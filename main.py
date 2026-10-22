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

WINDOW = "Invisible Box"


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
    if not Path(config.MODEL_PATH).exists():
        sys.exit(f"Modelo não encontrado em {config.MODEL_PATH}. Rode: python download_models.py")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir a câmera {args.camera}.")

    tracker = HandTracker()
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

            # 1) Calibração do fundo
            if not background.ready:
                background.add_calibration_frame(frame)
                view = frame.copy()
                overlay.draw_calibration(view, background.progress)
                if background.ready:
                    controller.background_ready()
                cv2.imshow(WINDOW, view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue

            # 2) Visão: mãos no frame ORIGINAL e máscara da pessoa
            hands = tracker.detect(frame)
            person = background.person_mask(frame)

            # 3) Gestos -> estado da janela
            controller.update(hands, now, dt)

            # 4) Efeito
            view = frame
            if controller.box is not None:
                box_mask = controller.box.mask(frame.shape, now)
                view = glass.render(frame, background.image(), person, box_mask,
                                    controller.box.bbox(), now)
            else:
                view = frame.copy()

            background.update(frame, person)

            # 5) Interface
            overlay.draw_box(view, controller, now)
            overlay.draw_preview(view, controller)
            overlay.draw_hands(view, controller, show_hands)
            if show_debug:
                overlay.debug_panel(view, background.image(), person)
            overlay.draw_hud(view, controller, fps)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                background.reset()
                controller.recalibrate()
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
