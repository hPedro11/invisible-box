"""Efeito de vidro transparente (Lab04: blur, Sobel e Canny; Lab05: máscaras).

Ideia: dentro da janela, a pessoa é trocada pelo FUNDO aprendido, e só sobra a
refração, como uma escultura de vidro. Montamos um "relevo" com duas partes:

- borda: a distância até o contorno (distanceTransform) vira uma rampa
  arredondada de GLASS_BEVEL px -> um "tubo" de vidro em volta do corpo;
- linhas internas: as bordas da própria pessoa (Canny), engrossadas e
  borradas -> dedos, braços e dobras da roupa viram filetes de vidro.

O miolo do corpo fica plano: ali o fundo aparece limpo, sem distorção.
O gradiente (Sobel) do relevo vira o mapa de deslocamento do cv2.remap. Cada
canal de cor é deslocado com força um pouco diferente (aberração cromática), e
um reflexo de luz vinda de cima/esquerda dá volume ao vidro.
"""
import cv2
import numpy as np

from . import config

# Para uma rampa de inclinação s, o Sobel 5x5 responde 128*s.
_SOBEL5_GAIN = 128.0
_LIGHT = np.array([0.6, 0.8], np.float32)   # luz vindo de cima e da esquerda


def _odd(v):
    return max(int(v), 1) | 1


def _ellipse(size):
    size = max(int(size), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class GlassEffect:
    def __init__(self):
        self._grid_cache = {}

    def _grid(self, h, w):
        key = (h, w)
        if key not in self._grid_cache:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_cache = {key: (gx, gy)}  # guarda só o último tamanho
        return self._grid_cache[key]

    @staticmethod
    def _long_edges(img, min_length):
        """Bordas da imagem (Canny) mantendo só as linhas com comprimento >= min_length px."""
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        lo, hi = config.GLASS_CANNY
        edges = cv2.Canny(gray, lo, hi)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # o contorno de uma linha aberta dá a volta nela: perímetro ~ 2x o comprimento
        keep = [c for c in contours if cv2.arcLength(c, False) >= min_length]
        canvas = np.zeros_like(edges)
        cv2.drawContours(canvas, keep, -1, 255, 1, cv2.LINE_AA)
        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)

    def render(self, frame, background, person_mask, box_mask, box_bbox, t):
        """Devolve o frame com a janela aplicada.

        Fora da janela: imagem normal. Dentro: onde não há pessoa, imagem ao vivo;
        onde há pessoa, o fundo refratado (a pessoa fica transparente).
        """
        H, W = frame.shape[:2]
        bevel = max(float(config.GLASS_BEVEL), 1.0)
        kr = _odd(config.GLASS_RIDGE_WIDTH)
        pad = int(config.GLASS_STRENGTH * (1 + config.GLASS_CHROMA) + bevel + kr)
        x, y, bw, bh = box_bbox
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + bw, W), min(y + bh, H)
        if x1 <= x0 or y1 <= y0:
            return frame.copy()

        # Onde não há pessoa a janela mostra a imagem ao vivo, ou seja, nada muda.
        # Então só processamos o retângulo em volta da pessoa DENTRO da janela.
        inside = cv2.bitwise_and(person_mask[y0:y1, x0:x1], box_mask[y0:y1, x0:x1])   # Lab05
        px, py, pw, ph = cv2.boundingRect(inside)
        if pw == 0 or ph == 0:
            return frame.copy()
        x0, y0, x1, y1 = (max(x0 + px - pad, 0), max(y0 + py - pad, 0),
                          min(x0 + px + pw + pad, W), min(y0 + py + ph + pad, H))

        f = frame[y0:y1, x0:x1]
        bg = background[y0:y1, x0:x1]
        boxm = box_mask[y0:y1, x0:x1]
        # o relevo usa a silhueta INTEIRA; a janela só recorta no final. Assim a
        # borda da janela corta o corpo "seco", sem virar um tubo de vidro.
        person = person_mask[y0:y1, x0:x1]

        # borda: rampa arredondada (0 no contorno -> 1 a GLASS_BEVEL px para dentro)
        dist = cv2.distanceTransform(person, cv2.DIST_L2, 3)
        ramp = np.minimum(dist / bevel, 1.0)
        relief = 1.0 - (1.0 - ramp) ** 2
        relief = cv2.GaussianBlur(relief, (5, 5), 0)

        # linhas internas: só os contornos LONGOS da pessoa (mão, dedos, braço, rosto).
        # Canny em meia resolução + filtro por comprimento (Lab05: contornos).
        if config.GLASS_RIDGES > 0:
            lines = self._long_edges(f, config.GLASS_RIDGE_MIN_LENGTH)
            core = cv2.erode(person, _ellipse(bevel))
            lines = cv2.bitwise_and(lines, core)
            ridges = cv2.GaussianBlur(lines.astype(np.float32) / 255.0, (kr, kr), 0)
            ridges = np.minimum(ridges * (kr / 4.0), 1.0)
            relief = cv2.scaleAdd(ridges, float(config.GLASS_RIDGES), relief)

        # inclinação do relevo; ~1 na borda do contorno
        gain = bevel / 2.0 / _SOBEL5_GAIN
        gx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=5, scale=gain)
        gy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=5, scale=gain)

        grid_x, grid_y = self._grid(*relief.shape)
        # ondulação animada dentro da silhueta; posição absoluta -> não "escorrega" com a janela.
        # sen(y) * cos(x) é separável: calcula 1 linha e 1 coluna e multiplica.
        body = (person > 0).astype(np.float32)
        wave_y = np.sin((np.arange(y0, y1, dtype=np.float32) * 0.05 + t * 4.0))[:, None]
        wave_x = np.cos((np.arange(x0, x1, dtype=np.float32) * 0.04 + t * 3.0))[None, :]
        shimmer = (wave_y * wave_x) * body * config.GLASS_SHIMMER
        base_x, base_y = grid_x + shimmer, grid_y + shimmer

        # um remap por canal, com força um pouco diferente: franjas coloridas
        channels = []
        for ch, k in zip(cv2.split(bg), (1 + config.GLASS_CHROMA, 1.0, 1 - config.GLASS_CHROMA)):  # B, G, R
            s = config.GLASS_STRENGTH * k
            channels.append(cv2.remap(ch, cv2.scaleAdd(gx, s, base_x), cv2.scaleAdd(gy, s, base_y),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        refracted = cv2.merge(channels)

        # luz: o lado do vidro virado para a luz brilha, o oposto escurece um pouco
        facing = cv2.scaleAdd(gx, float(_LIGHT[0]), gy * _LIGHT[1])
        rim = np.clip(cv2.magnitude(gx, gy) - 0.35, 0.0, 1.0)
        glint = np.clip(facing, 0.0, 1.0)
        glint = np.minimum((glint * glint + 0.35 * rim) * np.float32(config.GLASS_HIGHLIGHT), 1.0)
        shade = np.clip(-facing, 0.0, 1.0) * np.float32(0.35 * config.GLASS_HIGHLIGHT)
        tint = np.empty_like(refracted)
        tint[:] = config.GLASS_TINT
        glass = cv2.blendLinear(refracted, tint, 1.0 - glint, glint)
        glass = cv2.blendLinear(glass, np.zeros_like(glass), 1.0 - shade, shade)

        # mistura: vidro onde há pessoa (cobre um pouco além da borda), imagem ao vivo no resto.
        # blendLinear faz a média ponderada em C++, bem mais rápido que float no NumPy.
        alpha = cv2.GaussianBlur(cv2.dilate(person, _ellipse(5)).astype(np.float32) / 255.0, (7, 7), 0)
        mixed = cv2.blendLinear(f, glass, 1.0 - alpha, alpha)

        out = frame.copy()
        np.copyto(out[y0:y1, x0:x1], mixed, where=(boxm > 0)[..., None])
        return out
