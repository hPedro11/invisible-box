"""O processamento de um frame, compartilhado por main.py e calibrar.py.

Câmera -> flip -> mãos e pessoas (MediaPipe, em cópia reduzida) -> fundo ->
gestos -> efeito de vidro -> interface.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import config, overlay
from .background import BackgroundModel
from .controller import BoxController
from .glass import GlassEffect
from .hand_tracker import HandTracker
from .person_segmenter import PersonSegmenter



def open_camera(index, width, height):
    """Abre a câmera pedindo width x height e devolve (cap, (largura, altura) reais).

    Se a câmera não tiver o modo pedido, ela escolhe outro (ex.: pedindo
    960x540 muitas webcams entregam 640x480). Usamos o tamanho que ela
    entregar, sem esticar: esticar deixava a imagem borrada e distorcida.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        sys.exit(f"Não foi possível abrir a câmera {index}.")
    h, w = frame.shape[:2]
    pedido = f" (pedido {width}x{height})" if (w, h) != (width, height) else ""
    print(f"Câmera {index}: {w}x{h}{pedido}")
    return cap, (w, h)


class Pipeline:
    def __init__(self, size, mirror=True, show_debug=False):
        w, h = size
        self.tracker = HandTracker()
        self.segmenter = PersonSegmenter()
        self.background = BackgroundModel()
        self.glass = GlassEffect()
        self.controller = BoxController(w, h)
        self.mirror = mirror
        self.show_debug, self.show_hands = show_debug, True
        self.person = None
        self.now = time.perf_counter()
        self.fps = 0.0
        self._last = self.now
        # a segmentação roda numa thread em paralelo com as mãos (o MediaPipe
        # libera o Python enquanto calcula, então as duas andam juntas)
        self._vision = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentacao")
        self._frame_i = 0
        self._saw_hands = False

    def step(self, frame):
        """Processa um frame da câmera e devolve a imagem pronta para mostrar."""
        if self.mirror:
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        dt = min(now - self._last, 0.1)
        self._last, self.now = now, now
        self.fps = self.fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1

        # 1) Visão: mãos e pessoas ao mesmo tempo (o MediaPipe recebe cópias reduzidas).
        # Sem mãos na tela, procurar mãos é o mais caro (o modelo varre a imagem
        # toda), então nesse caso procura só a cada HAND_IDLE_EVERY frames.
        seg = self._vision.submit(self.segmenter.probability, frame)
        self._frame_i += 1
        if self._saw_hands or self._frame_i % max(config.HAND_IDLE_EVERY, 1) == 0:
            hands = self.tracker.detect(frame)
            self._saw_hands = bool(hands)
        else:
            hands = []
        prob = seg.result()
        self.person = person = self.background.person_mask(prob, frame.shape)

        # 2) Fundo: aprende tudo que não é pessoa (funciona com gente na cena)
        background = self.background.update(frame, person, prob)

        # 3) Gestos -> estado da janela
        controller = self.controller
        controller.update(hands, now, dt)

        # 4) Efeito
        missing = 0.0
        if controller.box is not None:
            box_mask = controller.box.mask(frame.shape, now)
            view = self.glass.render(frame, background, person, box_mask, controller.box.bbox(), now)
            missing = self.background.missing_fraction(box_mask)
        else:
            view = frame.copy()

        # 5) Interface
        overlay.draw_box(view, controller, now)
        if missing > config.BG_MISSING_WARN:
            overlay.draw_missing_notice(view, controller.box, missing)
        overlay.draw_preview(view, controller)
        overlay.draw_hands(view, controller, self.show_hands)
        if self.show_debug:
            overlay.debug_panel(view, background, person, self.background.unknown_mask())
        overlay.draw_hud(view, controller, self.fps, self.background.known_fraction)
        return view

    def handle_key(self, key):
        """Teclas comuns. Devolve False quando é para sair."""
        if key in (27, ord("q")):
            return False
        if key == ord("r"):
            self.background.reset()
        elif key == ord("n"):
            self.controller.new_box()
        elif key == ord("d"):
            self.show_debug = not self.show_debug
        elif key == ord("h"):
            self.show_hands = not self.show_hands
        return True

    def close(self):
        self._vision.shutdown(wait=True)
        self.tracker.close()
        self.segmenter.close()
        self.background.close()
