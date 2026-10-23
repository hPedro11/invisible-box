"""Parâmetros do Invisible Box. Ajuste aqui sem mexer no resto do código."""

# --- Câmera ---------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960          # resolução de trabalho (mantém o FPS alto)
FRAME_HEIGHT = 540
MIRROR = True              # espelha a imagem (fica natural, como um espelho)

# --- MediaPipe: mãos (Lab13) ---------------------------------------------
MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# --- MediaPipe: segmentação de pessoas ------------------------------------
# selfie_segmenter: ~6 ms por frame, bordas um pouco serrilhadas (padrão).
# selfie_multiclass_256x256: bordas melhores (cabelo, roupa), mas ~90 ms por frame.
SEGMENTER_PATH = "models/selfie_segmenter.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
SEG_THRESHOLD = 0.5        # probabilidade mínima para o pixel ser "pessoa" (Lab02)
SEG_SMOOTHING = 0.4        # média móvel da probabilidade entre frames (0 = sem suavizar)

# --- Gestos ---------------------------------------------------------------
# distância polegar-indicador dividida pelo tamanho da mão (pulso -> base do médio)
PINCH_ON = 0.30            # abaixo disso a pinça FECHA
PINCH_OFF = 0.45           # acima disso a pinça ABRE (histerese, como no Canny)
PINCH_REQUIRE_OPEN_FINGERS = True   # pinça = "OK" (médio/anelar/mínimo não todos dobrados)
POINTING_PINCH_WINDOW = 0.6  # s: pinça também vale se o indicador estava esticado há pouco
POINT_SMOOTHING = 0.5      # média móvel exponencial da posição (0 = sem suavizar)
TAP_MAX_SECONDS = 0.6      # pinça rápida: fechar e abrir em até este tempo salva a janela
TAP_LOOKBACK = 0.25        # s: os cantos salvos são os de antes do dedo descer para a pinça
RESET_CONFIRM_FRAMES = 30  # frames com as duas mãos abertas para recriar a janela

# --- Fundo aprendido com pessoas na cena (Lab07 / Lab06) --------------------
PERSON_DILATE = 21         # px: margem em volta das pessoas que NÃO entra no fundo
BG_LEARNING_RATE = 0.05    # quão rápido o fundo se adapta onde não há pessoa
BG_CONFIRM_FRAMES = 6      # frames seguidos sem pessoa até o pixel entrar no fundo
BG_MISSING_WARN = 0.03     # avisa se mais que 3% da janela ainda não tem fundo conhecido
BG_INPAINT_EVERY = 10      # frames entre os preenchimentos das áreas nunca vistas
BG_INPAINT_SCALE = 0.25    # escala do inpaint (menor = mais rápido)
MORPH_OPEN_KERNEL = 5      # remove ruído pequeno da máscara da pessoa
MORPH_CLOSE_KERNEL = 9     # fecha buracos pequenos

# --- Efeito de vidro transparente (Lab04) -----------------------------------
GLASS_BEVEL = 16           # px: largura da borda de vidro (o "tubo" do contorno)
GLASS_RIDGES = 0.7         # força das linhas de vidro internas (dedos, braços, rosto)
GLASS_RIDGE_WIDTH = 11     # largura das linhas internas (ímpar)
GLASS_RIDGE_MIN_LENGTH = 90  # px: linhas internas mais curtas que isso são ignoradas
GLASS_CANNY = (40, 110)    # limiares do Canny que acha as linhas internas
GLASS_STRENGTH = 26.0      # deslocamento máximo (px) da refração
GLASS_CHROMA = 0.35        # separação de cores (aberração cromática) nas bordas
GLASS_SHIMMER = 1.5        # ondulação interna (px), animada no tempo
GLASS_HIGHLIGHT = 0.55     # intensidade do brilho (reflexo da luz no vidro)
GLASS_TINT = (255, 235, 215)  # BGR do reflexo (azul bem claro)

# --- Janela ---------------------------------------------------------------
MIN_BOX_SIZE = 60
CORNER_GRAB_RADIUS = 40    # px: pinça perto de um canto -> redimensionar
GRAB_MARGIN = 15           # px de tolerância para "dentro da janela"
FRICTION = 0.92            # perda de velocidade por frame (60 fps de referência)
BOUNCE = 0.7               # energia mantida ao quicar na borda
THROW_MIN_SPEED = 250.0    # px/s mínimo para arremessar ao soltar
STOP_SPEED = 20.0          # px/s abaixo disso a janela para
WOBBLE_AMPLITUDE = 6.0     # px de ondulação das bordas quando a janela se move
