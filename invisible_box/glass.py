"""Efeito de vidro líquido (Lab04: GaussianBlur + Sobel, Lab05: máscaras).

Ideia: a máscara da pessoa é borrada, virando uma "colina" suave. O gradiente
dessa colina (Sobel) é quase zero no meio do corpo e forte nas bordas. Usamos
esse gradiente como mapa de deslocamento no cv2.remap: o fundo atrás da pessoa
aparece quase intacto no centro e se curva nas bordas, como uma lente.
"""
import cv2
import numpy as np

from . import config

# Para uma rampa de inclinação s, o Sobel 5x5 responde 128*s. Dividindo por 128
# temos a inclinação real; como a borda borrada tem inclinação ~1/GLASS_BLUR,
# multiplicar por GLASS_BLUR deixa o valor ~1 na borda.
_SOBEL5_GAIN = 128.0


class GlassEffect:
    def __init__(self):
        self._grid_cache = {}

    def _grid(self, h, w):
        key = (h, w)
        if key not in self._grid_cache:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_cache = {key: (gx, gy)}  # guarda só o último tamanho
        return self._grid_cache[key]

    def render(self, frame, background, person_mask, box_mask, box_bbox, t):
        """Devolve o frame com a janela aplicada.

        Fora da janela: imagem normal. Dentro: onde não há pessoa, imagem ao vivo;
        onde há pessoa, o fundo refratado com brilho nas bordas.
        """
        H, W = frame.shape[:2]
        pad = int(config.GLASS_STRENGTH + config.GLASS_BLUR)
        x, y, bw, bh = box_bbox
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, W), min(y + bh + pad, H)
        if x1 <= x0 or y1 <= y0:
            return frame

        # Trabalha só no recorte da janela: bem mais rápido que a imagem inteira.
        f = frame[y0:y1, x0:x1].astype(np.float32)
        bg = background[y0:y1, x0:x1]
        boxm = box_mask[y0:y1, x0:x1]
        inside = cv2.bitwise_and(person_mask[y0:y1, x0:x1], boxm)   # Lab05

        k = config.GLASS_BLUR | 1
        soft = cv2.GaussianBlur(inside.astype(np.float32) / 255.0, (k, k), 0)

        gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=5) / _SOBEL5_GAIN * k
        gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=5) / _SOBEL5_GAIN * k

        grid_x, grid_y = self._grid(*soft.shape)
        # ondulação animada dentro da silhueta (posição absoluta -> não "escorrega" com a janela)
        ax, ay = grid_x + x0, grid_y + y0
        wave = np.sin(ay * 0.05 + t * 4.0) * np.cos(ax * 0.04 + t * 3.0)
        shimmer = wave * soft * config.GLASS_SHIMMER

        map_x = grid_x + gx * config.GLASS_STRENGTH + shimmer
        map_y = grid_y + gy * config.GLASS_STRENGTH + shimmer
        refracted = cv2.remap(bg, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)

        # brilho nas bordas: onde o gradiente é forte
        edge = np.sqrt(gx * gx + gy * gy)
        glint = np.clip((edge - 0.25) * 1.6, 0.0, 1.0) ** 2 * config.GLASS_HIGHLIGHT
        tint = np.array(config.GLASS_TINT, dtype=np.float32)
        glass = refracted * (1.0 - glint[..., None]) + tint * glint[..., None]

        # mistura: vidro onde há pessoa (máscara suave), imagem ao vivo no resto
        alpha = np.clip(soft * 1.8, 0.0, 1.0)[..., None]
        mixed = f * (1.0 - alpha) + glass * alpha

        out = frame.copy()
        region = out[y0:y1, x0:x1]
        sel = boxm > 0
        region[sel] = np.clip(mixed[sel], 0, 255).astype(np.uint8)
        return out
