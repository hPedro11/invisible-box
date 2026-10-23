"""Fundo da cena aprendido mesmo com pessoas na frente (Lab07 + Lab06).

Não existe mais a etapa "saia da frente da câmera". A cada frame, a segmentação
do MediaPipe diz onde estão as pessoas; tudo que NÃO é pessoa (com uma margem)
entra no fundo. Quando alguém se mexe, o pedaço de cenário que estava atrás
aparece e é aprendido. O que nunca foi visto (alguém parado o tempo todo) é
preenchido com inpaint a partir dos vizinhos até ser visto de verdade.
"""
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
        self._margin_k = _ellipse(config.PERSON_DILATE)
        self.reset()

    def reset(self):
        self._bg = None       # float32, média do que já foi visto sem pessoa
        self._known = None    # uint8 0/255: pixels que já foram vistos sem pessoa
        self._fill = None     # cache do inpaint das áreas nunca vistas
        self._image = None    # fundo final (uint8) usado pelo efeito
        self._frames = 0

    @property
    def ready(self):
        return self._image is not None

    @property
    def known_fraction(self):
        return 0.0 if self._known is None else float(np.count_nonzero(self._known)) / self._known.size

    def image(self):
        return self._image

    def person_mask(self, prob):
        """Máscara binária (0/255) das pessoas a partir da probabilidade do segmentador."""
        mask = (prob > config.SEG_THRESHOLD).astype(np.uint8) * 255   # Lab02: limiarização
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)    # Lab06: tira pontinhos
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)  # Lab06: fecha buracos

    def update(self, frame, person_mask):
        """Aprende o fundo só onde não há pessoa (com margem em volta delas)."""
        free = cv2.bitwise_not(cv2.dilate(person_mask, self._margin_k))   # Lab05/Lab06
        if self._bg is None:
            self._bg = frame.astype(np.float32)
            self._known = np.zeros(frame.shape[:2], np.uint8)

        # pixel visto pela primeira vez: copia direto; os outros: média que acompanha a luz
        new = cv2.bitwise_and(free, cv2.bitwise_not(self._known))
        np.copyto(self._bg, frame, where=(new > 0)[..., None], casting="unsafe")
        cv2.accumulateWeighted(frame, self._bg, config.BG_LEARNING_RATE, mask=free)  # Lab07
        cv2.bitwise_or(self._known, free, dst=self._known)

        image = cv2.convertScaleAbs(self._bg)
        unknown = self._known == 0
        if unknown.any():
            if self._fill is None or self._frames % config.BG_INPAINT_EVERY == 0:
                self._fill = self._inpaint(image, unknown)
            np.copyto(image, self._fill, where=unknown[..., None])
        self._image = image
        self._frames += 1

    @staticmethod
    def _inpaint(image, unknown):
        """Preenche as áreas nunca vistas com os vizinhos (em escala menor, para ser rápido)."""
        h, w = unknown.shape
        s = config.BG_INPAINT_SCALE
        small = cv2.resize(image, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(unknown.astype(np.uint8) * 255, small.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST)
        small_mask = cv2.dilate(small_mask, _ellipse(3))
        filled = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
        return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
