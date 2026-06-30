import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import layers, regularizers

st.set_page_config(
    page_title="Classificação TI-RADS de Nódulos Tireoidianos",
    layout="wide",
)

MODEL_PATH = st.secrets["MODELO"]
#MODEL_PATH = "thyroid_cancer_model.h5"
IMG_SIZE = (224, 224)
DISPLAY_WIDTH = 250  # <- controla o tamanho da imagem exibida na tela

# Classes TI-RADS "candidatas", na ordem que o LabelEncoder usaria
# (ordena os rótulos originais de forma crescente: 2, 3, 4, 5).
# ATENÇÃO: se alguma pasta (ex: "3") não tiver imagens suficientes/nenhuma
# no treino, o LabelEncoder não vai gerar essa classe, e o modelo terá
# menos de 4 saídas. Por isso a lista efetiva é ajustada dinamicamente
# logo após o carregamento do modelo (ver `ajustar_classes_tirads`).
CLASSES_TIRADS_CANDIDATAS = ["2", "3", "4", "5"]
CLASSES_TIRADS = CLASSES_TIRADS_CANDIDATAS  # pode ser sobrescrito após carregar o modelo

# Estilo de exibição por categoria TI-RADS (2 e 3 = menor suspeita, 4 e 5 = maior suspeita)
ESTILO_TIRADS = {
    "2": "success",
    "3": "success",
    "4": "warning",
    "5": "error",
}

class Avg2MaxPooling(layers.Layer):

    def __init__(
        self,
        pool_size=3,
        strides=2,
        padding="same",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.avg_pool = layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )

        self.max_pool = layers.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )

        self.bn = layers.BatchNormalization()

    def call(self, inputs):

        x_avg = self.avg_pool(inputs)
        x_max = self.max_pool(inputs)

        x = x_avg - 2 * x_max

        return self.bn(x)

    def get_config(self):

        config = super().get_config()

        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })

        return config

class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(
            max(channels // self.ratio, 1),
            activation="swish"
        )

        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid"
        )
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

class DepthwiseSeparableConv(layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        se_ratio=16,
        reg=0.001,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.reg = reg

        self.depthwise = layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding="same",
            depthwise_regularizer=regularizers.l2(reg)
        )

        self.pointwise = layers.Conv2D(
            filters,
            1,
            padding="same",
            kernel_regularizer=regularizers.l2(reg)
        )

        self.bn = layers.BatchNormalization()

        self.se = SEBlock(se_ratio)

    def call(self, inputs):

        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        x = tf.nn.swish(x)
        x = self.se(x)

        return x

    def get_config(self):

        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "reg": self.reg
        })

        return config


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "Avg2MaxPooling": Avg2MaxPooling,
            "SEBlock": SEBlock,
            "DepthwiseSeparableConv": DepthwiseSeparableConv
        }
    )

    return model

def ajustar_classes_tirads(model):
    """
    Ajusta a lista global CLASSES_TIRADS para ter o mesmo tamanho da saída
    real do modelo. Isso evita KeyError quando o modelo foi treinado com
    menos classes do que o esperado (ex.: pasta "3" sem imagens no treino).

    OBS: isso é uma salvaguarda. Se uma categoria intermediária (ex: "3")
    estiver ausente, o ideal é confirmar no notebook de treino quais
    classes o LabelEncoder de fato gerou (`label_encoder.classes_`) e
    ajustar essa lista manualmente para refletir a ordem real.
    """
    global CLASSES_TIRADS
    n_saidas = model.output_shape[-1]

    if n_saidas != len(CLASSES_TIRADS_CANDIDATAS):
        st.warning(
            f"O modelo tem {n_saidas} saída(s), mas eram esperadas "
            f"{len(CLASSES_TIRADS_CANDIDATAS)} categorias TI-RADS "
            f"({', '.join(CLASSES_TIRADS_CANDIDATAS)}). "
            "Uma ou mais categorias podem não ter sido usadas no treino "
            "(ex.: pasta sem imagens). Verifique `label_encoder.classes_` "
            "no notebook para confirmar o mapeamento correto."
        )
        CLASSES_TIRADS = CLASSES_TIRADS_CANDIDATAS[:n_saidas]
    else:
        CLASSES_TIRADS = CLASSES_TIRADS_CANDIDATAS

def corrigir_orientacao(image):
    """
    Corrige a orientação da imagem com base nos metadados EXIF.

    Fotos tiradas de celular costumam salvar a orientação correta apenas
    nos metadados EXIF, mantendo os pixels "crus" rotacionados. O
    PIL.Image.open() NÃO aplica essa rotação automaticamente, então sem
    essa correção a imagem pode ser enviada ao modelo rotacionada
    (90°/180°) mesmo aparecendo "em pé" na tela -- o que o modelo nunca
    viu no treino (imagens de ultrassom já vinham no enquadramento
    correto), causando classificações incorretas.
    """
    return ImageOps.exif_transpose(image)

def preprocess_image(image):
    image = corrigir_orientacao(image)
    image = image.convert("RGB")
    # NEAREST para reproduzir o mesmo comportamento do ImageDataGenerator
    # (Keras `load_img`) usado no treino do modelo, que usa interpolação
    # 'nearest' por padrão ao redimensionar para target_size.
    image = image.resize(IMG_SIZE, Image.NEAREST)
    img = np.array(image).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def gerar_thumbnail(image, size=(DISPLAY_WIDTH, DISPLAY_WIDTH)):
    """
    Recorta a imagem no centro para a proporção desejada
    e redimensiona para um tamanho fixo, garantindo que
    todas as miniaturas exibidas tenham exatamente o
    mesmo tamanho, independente da imagem original.
    """

    image = corrigir_orientacao(image)
    image = image.convert("RGB")

    target_w, target_h = size
    target_ratio = target_w / target_h

    w, h = image.size
    current_ratio = w / h

    if current_ratio > target_ratio:
        # imagem mais larga que o alvo -> recorta as laterais
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        # imagem mais alta que o alvo -> recorta topo/base
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)

    image_cropped = image.crop(box)
    thumbnail = image_cropped.resize(size, Image.LANCZOS)

    return thumbnail


def classificar_imagem(model, image):

    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)[0]  # vetor com 4 probabilidades (softmax)

    idx_predito = int(np.argmax(prediction))
    classe = CLASSES_TIRADS[idx_predito]
    confianca = float(prediction[idx_predito])

    # Dicionário {categoria: probabilidade} para exibir todas as probabilidades depois
    probabilidades = {
        cat: float(prob) for cat, prob in zip(CLASSES_TIRADS, prediction)
    }

    return classe, confianca, probabilidades

st.title("Classificação TI-RADS de Nódulos Tireoidianos")
st.caption("O modelo classifica o nódulo em uma das categorias TI-RADS: 2, 3, 4 ou 5.")
try:
    model = load_model()
    ajustar_classes_tirads(model)
except Exception as e:
    st.error("Erro ao carregar o modelo")
    st.code(str(e))
    st.stop()

# Lista de imagens (em sessão) usada tanto para upload quanto para câmera
if "imagens" not in st.session_state:
    st.session_state.imagens = []  # cada item: PIL.Image

opcao = st.radio(
    "Escolha uma opção",
    ["Upload", "Câmera"],
    horizontal=True
)

if opcao == "Upload":

    uploaded_files = st.file_uploader(
        "Selecione uma ou mais imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.imagens = [
            Image.open(f) for f in uploaded_files
        ]
else:

    captured = st.camera_input("Capture uma imagem")
    col_a, col_b = st.columns(2)
    with col_a:
        if captured and st.button("➕ Adicionar foto capturada"):
            st.session_state.imagens.append(Image.open(captured))
            st.success("Foto adicionada! Capture outra ou avance para a análise.")
    with col_b:
        if st.button("🗑️ Limpar todas as fotos"):
            st.session_state.imagens = []
imagens = st.session_state.imagens

if imagens:
    st.markdown(f"**{len(imagens)} imagem(ns) selecionada(s)**")
    n_cols = 4
    cols = st.columns(n_cols)
    resultados = []
    with st.spinner("Classificando imagens..."):
        for idx, image in enumerate(imagens):
            classe, confianca, probabilidades = classificar_imagem(model, image)
            resultados.append((image, classe, confianca, probabilidades))

    for idx, (image, classe, confianca, probabilidades) in enumerate(resultados):
        col = cols[idx % n_cols]
        with col:
            thumb = gerar_thumbnail(image)
            st.image(
                thumb,
                caption=f"Imagem {idx + 1}"
            )

            estilo = ESTILO_TIRADS.get(classe, "info")
            if estilo == "error":
                st.error(f"TI-RADS {classe}")
            elif estilo == "warning":
                st.warning(f"TI-RADS {classe}")
            else:
                st.success(f"TI-RADS {classe}")

            st.progress(float(confianca))
            st.write(f"Confiança: {confianca * 100:.2f}%")

            with st.expander("Probabilidades por categoria"):
                for cat, prob in probabilidades.items():
                    st.write(f"TI-RADS {cat}: {prob * 100:.2f}%")
else:
    st.info("Nenhuma imagem selecionada ainda.")