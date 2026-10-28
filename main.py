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

from invisible_box import config
from invisible_box.pipeline import Pipeline, open_camera

WINDOW = "Invisible Box"


def parse_args():
    p = argparse.ArgumentParser(description="Invisible Box")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="largura pedida à câmera")
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="altura pedida à câmera")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--debug", action="store_true", help="mostra fundo e máscara")
    return p.parse_args()


def check_models():
    for path in (config.MODEL_PATH, config.SEGMENTER_PATH):
        if not Path(path).exists():
            sys.exit(f"Modelo não encontrado em {path}. Rode: python download_models.py")


def create_window(name, size):
    """Janela redimensionável que mantém a proporção da câmera."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, *size)


def window_closed(name):
    """True se o usuário fechou a janela no X (sem isso o loop continuava rodando)."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    args = parse_args()
    check_models()
    cap, size = open_camera(args.camera, args.width, args.height)
    pipe = Pipeline(size, mirror=config.MIRROR and not args.no_mirror, show_debug=args.debug)
    create_window(WINDOW, size)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler a câmera.")
                break
            view = pipe.step(frame)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if window_closed(WINDOW):          # fechou no X
                break
            if key == ord("s"):
                Path("capturas").mkdir(exist_ok=True)
                cv2.imwrite(f"capturas/invisible_box_{int(time.time())}.png", view)
            elif not pipe.handle_key(key):
                break
    finally:
        pipe.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
