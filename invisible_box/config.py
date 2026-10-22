"""Parâmetros do Invisible Box. Ajuste aqui sem mexer no resto do código."""

# --- Câmera ---------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960          # resolução de trabalho (mantém o FPS alto)
FRAME_HEIGHT = 540
MIRROR = True              # espelha a imagem (fica natural, como um espelho)

# --- MediaPipe (Lab13) ----------------------------------------------------
MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# --- Gestos ---------------------------------------------------------------
# distância polegar-indicador dividida pelo tamanho da mão (pulso -> base do médio)
PINCH_ON = 0.30            # abaixo disso a pinça FECHA
PINCH_OFF = 0.45           # acima disso a pinça ABRE (histerese, como no Canny)
PINCH_REQUIRE_OPEN_FINGERS = True   # pinça = "OK" (médio/anelar/mínimo não todos dobrados)
POINT_SMOOTHING = 0.5      # média móvel exponencial da posição da pinça (0 = sem suavizar)
DEFINE_CONFIRM_FRAMES = 8  # frames com as duas pinças fechadas para criar a janela (Lab12)
RESET_CONFIRM_FRAMES = 30  # frames com as duas mãos abertas para recriar a janela

# --- Fundo e máscara da pessoa (Lab07 / Lab06) ----------------------------
CALIBRATION_FRAMES = 45    # frames médios para o fundo inicial (cena sem ninguém)
BG_LEARNING_RATE = 0.01    # quão rápido o fundo se adapta (só onde NÃO há pessoa)
DIFF_THRESHOLD = 28        # limiar da diferença frame x fundo
MORPH_OPEN_KERNEL = 5      # remove ruído pequeno
MORPH_CLOSE_KERNEL = 15    # fecha buracos no corpo
MASK_DILATE = 2            # iterações de dilatação final

# --- Efeito de vidro (Lab04) ----------------------------------------------
GLASS_BLUR = 31            # suavização da máscara (ímpar). Maior = bordas mais "gordas"
GLASS_STRENGTH = 22.0      # deslocamento máximo (px) da refração nas bordas
GLASS_SHIMMER = 3.0        # ondulação interna (px), animada no tempo
GLASS_HIGHLIGHT = 0.55     # intensidade do brilho branco nas bordas
GLASS_TINT = (255, 230, 200)  # BGR do reflexo (azul bem claro)

# --- Janela ---------------------------------------------------------------
MIN_BOX_SIZE = 60
CORNER_GRAB_RADIUS = 40    # px: pinça perto de um canto -> redimensionar
GRAB_MARGIN = 15           # px de tolerância para "dentro da janela"
FRICTION = 0.92            # perda de velocidade por frame (60 fps de referência)
BOUNCE = 0.7               # energia mantida ao quicar na borda
THROW_MIN_SPEED = 250.0    # px/s mínimo para arremessar ao soltar
STOP_SPEED = 20.0          # px/s abaixo disso a janela para
WOBBLE_AMPLITUDE = 6.0     # px de ondulação das bordas quando a janela se move
