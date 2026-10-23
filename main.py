"""Invisible Box: janela de invisibilidade controlada por gestos.

Uso:
    python download_models.py      # uma vez
    python main.py                 # webcam padrão
    python main.py --camera 1      # outra câmera
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from invisible_box import config, overlay
from invisible_box.background import BackgroundModel
from invisible_box.controller import BoxController
from invisible_box.glass import GlassEffect
from invisible_box.hand_tracker import HandTracker
from invisible_box.person_segmenter import PersonSegmenter

WINDOW = "Invisible Box"
# a câmera nunca viu o fundo atrás de quem não saiu do lugar
MISSING_BG_WARNING = "Fundo atras de voce ainda nao foi visto: saia do quadro por 2 s"


def parse_args():
    p = argparse.ArgumentParser(description="Invisible Box")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--debug", action="store_true", help="mostra fundo e máscara")
    return p.parse_args()


def main():
    args = parse_args()
    for path in (config.MODEL_PATH, config.SEGMENTER_PATH):
        if not Path(path).exists():
            sys.exit(f"Modelo não encontrado em {path}. Rode: python download_models.py")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir a câmera {args.camera}.")

    tracker = HandTracker()
    segmenter = PersonSegmenter()
    background = BackgroundModel()
    glass = GlassEffect()
    controller = BoxController(args.width, args.height)
    show_debug, show_hands = args.debug, True
    mirror = config.MIRROR and not args.no_mirror

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler a câmera.")
                break
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            fps = fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1

            # 1) Visão: mãos e pessoas no frame ORIGINAL
            hands = tracker.detect(frame)
            person = background.person_mask(segmenter.probability(frame))

            # 2) Fundo: aprende tudo que não é pessoa (funciona com gente na cena)
            background.update(frame, person)

            # 3) Gestos -> estado da janela
            controller.update(hands, now, dt)

            # 4) Efeito
            warning = None
            if controller.box is not None:
                box_mask = controller.box.mask(frame.shape, now)
                view = glass.render(frame, background.image(), person, box_mask,
                                    controller.box.bbox(), now)
                if background.missing_fraction(box_mask) > config.BG_MISSING_WARN:
                    warning = MISSING_BG_WARNING
            else:
                view = frame.copy()

            # 5) Interface
            overlay.draw_box(view, controller, now)
            overlay.draw_preview(view, controller)
            overlay.draw_hands(view, controller, show_hands)
            if show_debug:
                overlay.debug_panel(view, background.image(), person, background.unknown_mask())
            overlay.draw_hud(view, controller, fps, background.known_fraction, warning)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                background.reset()
            elif key == ord("n"):
                controller.new_box()
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("s"):
                Path("capturas").mkdir(exist_ok=True)
                cv2.imwrite(f"capturas/invisible_box_{int(time.time())}.png", view)
    finally:
        tracker.close()
        segmenter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
