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
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        lo, hi = config.GLASS_CANNY
        edges = cv2.Canny(gray, lo, hi)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # o contorno de uma linha aberta dá a volta nela: perímetro ~ 2x o comprimento
        keep = [c for c in contours if cv2.arcLength(c, False) >= 2 * min_length]
        canvas = np.zeros_like(edges)
        cv2.drawContours(canvas, keep, -1, 255, 1, cv2.LINE_AA)
        return canvas

    def _relief_maps(self, f, person, boxm, k, origin, t):
        """Todos os mapas suaves do vidro, calculados em escala reduzida e ampliados.

        Relevo, inclinação, ondulação, brilho, sombra e transparência variam
        devagar, então calculá-los em GLASS_RELIEF_SCALE e ampliar dá o mesmo
        resultado visual com bem menos trabalho. Devolve, no tamanho do recorte:
        gx, gy (inclinação, ~1 na borda), shimmer (px), glint, shade, alpha (0..1).
        O alpha já vem recortado pela janela (`boxm`), com a borda suavizada.
        """
        h, w = person.shape
        rs = config.GLASS_RELIEF_SCALE
        unit = k * rs                         # px reduzidos por px de referência
        sw, sh = max(round(w * rs), 1), max(round(h * rs), 1)
        person_s = cv2.resize(person, (sw, sh), interpolation=cv2.INTER_NEAREST)
        bevel = max(config.GLASS_BEVEL * unit, 1.0)

        # borda: rampa arredondada (0 no contorno -> 1 a GLASS_BEVEL px para dentro)
        dist = cv2.distanceTransform(person_s, cv2.DIST_L2, 3)
        ramp = np.minimum(dist / bevel, 1.0)
        relief = 1.0 - (1.0 - ramp) ** 2
        relief = cv2.GaussianBlur(relief, (3, 3), 0)

        # linhas internas: só os contornos LONGOS da pessoa (mão, dedos, braço, rosto).
        # Canny + filtro por comprimento (Lab05: contornos).
        if config.GLASS_RIDGES > 0:
            f_s = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
            lines = self._long_edges(f_s, config.GLASS_RIDGE_MIN_LENGTH * unit)
            core = cv2.erode(person_s, _ellipse(bevel))
            lines = cv2.bitwise_and(lines, core)
            kr = _odd(config.GLASS_RIDGE_WIDTH * unit)
            ridges = cv2.GaussianBlur(lines.astype(np.float32) / 255.0, (kr, kr), 0)
            ridges = np.minimum(ridges * (kr / 4.0), 1.0)
            relief = cv2.scaleAdd(ridges, float(config.GLASS_RIDGES), relief)

        # inclinação do relevo; ~1 na borda do contorno
        gain = bevel / 2.0 / _SOBEL5_GAIN
        gx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=5, scale=gain)
        gy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=5, scale=gain)

        # ondulação animada dentro da silhueta; posição absoluta (em px de referência)
        # -> não "escorrega" com a janela. sen(y)*cos(x) é separável.
        x0, y0 = origin
        ys = (y0 + np.arange(sh, dtype=np.float32) / rs) / k
        xs = (x0 + np.arange(sw, dtype=np.float32) / rs) / k
        wave = np.outer(np.sin(ys * 0.05 + t * 4.0), np.cos(xs * 0.04 + t * 3.0)).astype(np.float32)
        shimmer = cv2.multiply(wave, person_s.astype(np.float32),
                               scale=config.GLASS_SHIMMER * k / 255.0)

        # luz: o lado do vidro virado para a luz brilha, o oposto escurece um pouco
        facing = cv2.addWeighted(gx, float(_LIGHT[0]), gy, float(_LIGHT[1]), 0.0)
        rim = np.clip(cv2.magnitude(gx, gy) - 0.35, 0.0, 1.0)
        glint = np.clip(facing, 0.0, 1.0)
        glint = np.minimum((glint * glint + 0.35 * rim) * np.float32(config.GLASS_HIGHLIGHT), 1.0)
        shade = np.clip(-facing, 0.0, 1.0) * np.float32(0.35 * config.GLASS_HIGHLIGHT)

        # transparência: cobre a pessoa e um pouco além da borda (esconde o "halo"),
        # só dentro da janela (Lab05: máscara x máscara)
        alpha = cv2.dilate(person_s, _ellipse(5 * unit)).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (_odd(7 * unit), _odd(7 * unit)), 0)
        box_s = cv2.resize(boxm, (sw, sh), interpolation=cv2.INTER_AREA)
        alpha = cv2.multiply(alpha, box_s.astype(np.float32), scale=1.0 / 255.0)

        maps = (gx, gy, shimmer, glint, shade, alpha)
        if (sw, sh) == (w, h):
            return maps
        return tuple(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in maps)

    def render(self, frame, background, person_mask, box_mask, box_bbox, t):
        """Devolve o frame com a janela aplicada.

        Fora da janela: imagem normal. Dentro: onde não há pessoa, imagem ao vivo;
        onde há pessoa, o fundo refratado (a pessoa fica transparente).
        """
        H, W = frame.shape[:2]
        k = W / config.GLASS_REF_WIDTH            # px de referência -> px desta imagem
        strength = config.GLASS_STRENGTH * k
        pad = int(strength * (1 + config.GLASS_CHROMA)
                  + (config.GLASS_BEVEL + config.GLASS_RIDGE_WIDTH) * k)
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
        gx, gy, shimmer, glint, shade, alpha = self._relief_maps(f, person, boxm, k, (x0, y0), t)

        grid_x, grid_y = self._grid(*person.shape)
        base_x, base_y = cv2.add(grid_x, shimmer), cv2.add(grid_y, shimmer)

        # um remap por canal, com força um pouco diferente: franjas coloridas
        channels = []
        for ch, c in zip(cv2.split(bg), (1 + config.GLASS_CHROMA, 1.0, 1 - config.GLASS_CHROMA)):  # B, G, R
            s = strength * c
            channels.append(cv2.remap(ch, cv2.scaleAdd(gx, s, base_x), cv2.scaleAdd(gy, s, base_y),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        refracted = cv2.merge(channels)

        # brilho (puxa para a cor do reflexo) e sombra (puxa para o preto).
        # blendLinear faz a média ponderada em C++, bem mais rápido que float no NumPy.
        tint = np.empty_like(refracted)
        tint[:] = config.GLASS_TINT
        glass = cv2.blendLinear(refracted, tint, 1.0 - glint, glint)
        glass = cv2.blendLinear(glass, np.zeros_like(glass), 1.0 - shade, shade)

        # mistura: vidro onde há pessoa dentro da janela, imagem ao vivo no resto.
        # Como alpha = 0 fora disso, dá para copiar o recorte inteiro de uma vez.
        out = frame.copy()
        out[y0:y1, x0:x1] = cv2.blendLinear(f, glass, 1.0 - alpha, alpha)
        return out
