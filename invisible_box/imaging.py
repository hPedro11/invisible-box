"""Utilidades de imagem compartilhadas pelos módulos de visão."""
import cv2

from . import config


def reduced_copy(frame, width=None):
    """Cópia reduzida do frame (mesma proporção) para os modelos do MediaPipe.

    Os modelos redimensionam a entrada para algo como 256x256 de qualquer
    jeito; mandar a imagem já pequena só economiza conversão e cópia.
    """
    width = width or config.PROCESS_WIDTH
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
