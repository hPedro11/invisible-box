"""Fundo da cena aprendido mesmo com pessoas na frente (Lab07 + Lab06).

Não existe mais a etapa "saia da frente da câmera". A cada frame, a segmentação
do MediaPipe diz onde estão as pessoas; tudo que NÃO é pessoa (com uma margem)
entra no fundo. Quando alguém se mexe, o pedaço de cenário que estava atrás
aparece e é aprendido. O que nunca foi visto (alguém parado o tempo todo) é
preenchido com inpaint a partir dos vizinhos até ser visto de verdade.

Exposição: a webcam clareia/escurece sozinha (escurece quando a pessoa sai e
só sobra parede branca). O fundo é guardado numa exposição de REFERÊNCIA; a
cada atualização medimos o ganho atual da câmera na parte visível do fundo e
aplicamos esse ganho ao fundo inteiro. Sem isso, o pedaço atrás da pessoa
ficava "congelado" na exposição antiga e aparecia como uma mancha cinza.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config


def _ellipse(size):
    size = max(int(size), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class BackgroundModel:
    def __init__(self):
        self._open_k = _ellipse(config.MORPH_OPEN_KERNEL)
        self._close_k = _ellipse(config.MORPH_CLOSE_KERNEL)
        # retângulo: a dilatação fica separável e bem mais rápida que com elipse
        self._margin_k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.PERSON_DILATE, config.PERSON_DILATE))
        self._executor = None   # thread do inpaint (criada só quando precisa)
        self.reset()

    def reset(self):
        self._bg = None       # float32, média do que já foi visto sem pessoa
        self._known = None    # uint8 0/255: pixels que já foram vistos sem pessoa
        self._streak = None   # uint8: frames seguidos em que o pixel está sem pessoa
        self._fill = None     # cache do inpaint das áreas nunca vistas
        self._pending = None  # inpaint rodando em paralelo
        self._image = None    # fundo final (uint8) usado pelo efeito
        self._frames = 0
        self._person_prev = None  # máscara anterior (para a histerese)
        self._gain = np.ones(3, np.float32)  # exposição atual / exposição de referência (B, G, R)

    @property
    def ready(self):
        return self._image is not None

    @property
    def known_fraction(self):
        return 0.0 if self._known is None else cv2.countNonZero(self._known) / self._known.size

    def image(self):
        return self._image

    @property
    def exposure_gain(self):
        """Ganho atual da câmera em relação à exposição de referência (B, G, R)."""
        return self._gain.copy()

    def unknown_mask(self):
        """255 onde o fundo nunca foi visto (preenchido por inpaint)."""
        return cv2.bitwise_not(self._known)

    def missing_fraction(self, region_mask):
        """Fração da região (ex.: a janela) cujo fundo ainda não foi visto."""
        total = cv2.countNonZero(region_mask)
        if self._known is None or total == 0:
            return 0.0
        return cv2.countNonZero(cv2.bitwise_and(region_mask, self.unknown_mask())) / total

    def person_mask(self, prob, shape):
        """Máscara binária (0/255) das pessoas no tamanho `shape` do frame.

        Limiar e morfologia rodam na resolução do segmentador (pequena, rápido);
        a ampliação é linear + novo limiar, o que deixa a borda lisa em vez de
        "em escadinha".

        Limiar duplo (histerese, como no Canny): o pixel vira pessoa acima de
        SEG_THRESHOLD, mas só deixa de ser abaixo de SEG_KEEP_THRESHOLD. A borda
        para de oscilar e regiões de pouca certeza (cabelo) continuam cobertas.
        """
        mask = (prob > config.SEG_THRESHOLD).astype(np.uint8) * 255   # Lab02: limiarização
        prev = self._person_prev
        if prev is not None and prev.shape == mask.shape:
            keep = (prob > config.SEG_KEEP_THRESHOLD).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(keep, prev))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)    # Lab06: tira pontinhos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)  # Lab06: fecha buracos
        self._person_prev = mask
        h, w = shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def update(self, frame, person_mask):
        """Aprende o fundo só onde não há pessoa (com margem em volta delas)."""
        if self._bg is None:
            self._bg = frame.astype(np.float32)
            self._known = np.zeros(frame.shape[:2], np.uint8)
            self._streak = np.zeros(frame.shape[:2], np.uint8)
        self._frames += 1

        # o fundo muda devagar: aprender a cada BG_UPDATE_EVERY frames basta
        if self._image is None or self._frames % config.BG_UPDATE_EVERY == 0:
            self._learn(frame, person_mask)
        return self._image

    def _learn(self, frame, person_mask):
        free = cv2.bitwise_not(cv2.dilate(person_mask, self._margin_k))   # Lab05/Lab06

        # só confia no pixel depois de alguns frames seguidos sem pessoa: se o
        # segmentador falhar em um frame, um pedaço da pessoa não vira "fundo".
        cv2.add(self._streak, 1, dst=self._streak, mask=free)
        self._streak[free == 0] = 0
        confirm = max(1, -(-config.BG_CONFIRM_FRAMES // config.BG_UPDATE_EVERY))   # em atualizações
        stable = cv2.compare(self._streak, confirm, cv2.CMP_GE)

        # ganho de exposição medido no fundo visível que já conhecíamos;
        # o frame entra no fundo convertido para a exposição de referência
        if config.BG_EXPOSURE_COMPENSATION:
            self._update_gain(frame, cv2.bitwise_and(stable, self._known))
            g = self._gain
            target = cv2.multiply(frame.astype(np.float32),
                                  (float(1.0 / g[0]), float(1.0 / g[1]), float(1.0 / g[2]), 0.0))
        else:
            target = frame

        # pixel confirmado pela primeira vez: copia direto (taxa 1.0); os outros:
        # média que acompanha a luz (Lab07)
        new = cv2.bitwise_and(stable, cv2.bitwise_not(self._known))
        cv2.accumulateWeighted(target, self._bg, 1.0, mask=new)
        cv2.accumulateWeighted(target, self._bg, config.BG_LEARNING_RATE, mask=stable)
        cv2.bitwise_or(self._known, stable, dst=self._known)

        # fundo mostrado = fundo de referência x ganho atual da câmera
        g = self._gain
        image = cv2.convertScaleAbs(cv2.multiply(self._bg, (float(g[0]), float(g[1]), float(g[2]), 0.0)))
        unknown = cv2.bitwise_not(self._known)
        if cv2.countNonZero(unknown):
            self._refresh_fill(image, unknown)
            cv2.copyTo(self._fill, unknown, image)
        else:
            self._fill = None
        self._image = image

    def _refresh_fill(self, image, unknown):
        """Mantém o "chute" das áreas nunca vistas atualizado.

        O inpaint em resolução boa leva dezenas de ms; ele roda numa thread em
        paralelo (o OpenCV libera o Python enquanto calcula) e o resultado entra
        quando fica pronto. Até lá, vale o chute anterior. Só o primeiro é
        calculado na hora, porque ainda não há nenhum.
        """
        if self._pending is not None and self._pending.done():
            fill = self._pending.result()
            self._pending = None
            if fill.shape == image.shape:
                self._fill = fill
        if self._fill is None or self._fill.shape != image.shape:
            self._fill = self._inpaint(image, unknown)
        elif self._pending is None and self._frames % config.BG_INPAINT_EVERY == 0:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
            self._pending = self._executor.submit(self._inpaint, image.copy(), unknown.copy())

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _update_gain(self, frame, visible):
        """Mede quanto a câmera clareou/escureceu: frame / fundo na parte visível.

        Usa versões pequenas (rápido) e ignora pixels estourados ou quase pretos,
        onde a razão não diz nada. Razão das somas por canal = robusto a ruído.
        """
        h, w = visible.shape
        sw = 160
        sh = max(round(h * sw / w), 1)
        f = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(self._bg, (sw, sh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(visible, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        m &= (f.min(axis=2) > 8) & (f.max(axis=2) < 247)
        if np.count_nonzero(m) < 200:          # pouco fundo visível: mantém o ganho
            return
        gain = f[m].sum(axis=0) / np.maximum(b[m].sum(axis=0), 1.0)
        gain = np.clip(gain, 0.3, 3.0).astype(np.float32)
        a = config.BG_GAIN_SMOOTHING
        self._gain = self._gain * a + gain * (1.0 - a)

    @staticmethod
    def _inpaint(image, unknown):
        """Preenche as áreas nunca vistas com os vizinhos (em escala menor, para ser rápido)."""
        h, w = unknown.shape
        sw = min(config.BG_INPAINT_WIDTH, w)
        sh = max(round(h * sw / w), 1)
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(unknown, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_mask = cv2.dilate(small_mask, _ellipse(3))
        filled = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
        return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
