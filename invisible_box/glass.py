"""Efeito de vidro transparente (Lab04: GaussianBlur + Sobel, Lab05: máscaras).

Ideia: dentro da janela, a pessoa é trocada pelo FUNDO aprendido, e só sobra a
refração, como uma escultura de vidro. Montamos um "relevo" com duas partes:

- a silhueta borrada (Lab04): sobe na borda do corpo -> contorno de vidro;
- os detalhes internos da própria pessoa (passa-alta da imagem em cinza):
  dedos, braços e dobras da roupa viram linhas de vidro por dentro.

O gradiente (Sobel) desse relevo vira o mapa de deslocamento do cv2.remap. Cada
canal de cor é deslocado com força um pouco diferente (aberração cromática),
o que dá as franjas coloridas das bordas.
"""
import cv2
import numpy as np

from . import config

# Para uma rampa de inclinação s, o Sobel 5x5 responde 128*s. Dividindo por 128
# temos a inclinação real; multiplicando pelo tamanho do blur da borda, o valor
# fica ~1 na borda da silhueta.
_SOBEL5_GAIN = 128.0
_DETAIL_BASE_BLUR = 21     # passa-alta = imagem - blur grande dela


def _odd(v):
    return max(int(v), 1) | 1


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
        onde há pessoa, o fundo refratado (a pessoa fica transparente).
        """
        H, W = frame.shape[:2]
        ke = _odd(config.GLASS_EDGE_BLUR)
        pad = int(config.GLASS_STRENGTH * (1 + config.GLASS_CHROMA) + ke + _DETAIL_BASE_BLUR)
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
        inside = cv2.bitwise_and(person_mask[y0:y1, x0:x1], boxm)
        soft = cv2.GaussianBlur(inside.astype(np.float32) / 255.0, (ke, ke), 0)

        # detalhes internos: passa-alta da imagem em cinza, só dentro da pessoa
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        kd = _odd(config.GLASS_DETAIL_BLUR)
        detail = (cv2.GaussianBlur(gray, (kd, kd), 0)
                  - cv2.GaussianBlur(gray, (_DETAIL_BASE_BLUR, _DETAIL_BASE_BLUR), 0))
        relief = soft * (1.0 + config.GLASS_DETAIL * detail)

        gain = ke / _SOBEL5_GAIN
        gx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=5, scale=gain)
        gy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=5, scale=gain)

        grid_x, grid_y = self._grid(*soft.shape)
        # ondulação animada dentro da silhueta; posição absoluta -> não "escorrega" com a janela.
        # sen(y) * cos(x) é separável: calcula 1 linha e 1 coluna e multiplica.
        wave_y = np.sin((np.arange(y0, y1, dtype=np.float32) * 0.05 + t * 4.0))[:, None]
        wave_x = np.cos((np.arange(x0, x1, dtype=np.float32) * 0.04 + t * 3.0))[None, :]
        shimmer = (wave_y * wave_x) * soft * config.GLASS_SHIMMER
        base_x, base_y = grid_x + shimmer, grid_y + shimmer

        # um remap por canal, com força um pouco diferente: franjas coloridas
        channels = []
        for ch, k in zip(cv2.split(bg), (1 + config.GLASS_CHROMA, 1.0, 1 - config.GLASS_CHROMA)):  # B, G, R
            s = config.GLASS_STRENGTH * k
            channels.append(cv2.remap(ch, cv2.scaleAdd(gx, s, base_x), cv2.scaleAdd(gy, s, base_y),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        refracted = cv2.merge(channels)

        # brilho onde o relevo é inclinado: contorno e linhas internas
        edge = cv2.magnitude(gx, gy)
        glint = np.clip((edge - 0.3) * 1.5, 0.0, 1.0)
        glint = glint * glint * np.float32(config.GLASS_HIGHLIGHT)
        tint = np.empty_like(refracted)
        tint[:] = config.GLASS_TINT
        glass = cv2.blendLinear(refracted, tint, 1.0 - glint, glint)

        # mistura: vidro onde há pessoa (cobre um pouco além da borda), imagem ao vivo no resto.
        # blendLinear faz a média ponderada em C++, bem mais rápido que float no NumPy.
        alpha = np.clip(soft * 2.5, 0.0, 1.0)
        mixed = cv2.blendLinear(f, glass, 1.0 - alpha, alpha)

        out = frame.copy()
        np.copyto(out[y0:y1, x0:x1], mixed, where=(boxm > 0)[..., None])
        return out
