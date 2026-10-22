"""Modelo de fundo da cena inteira e máscara da pessoa (Lab07 + Lab06).

O fundo é da cena INTEIRA (não só da janela), então a janela pode ser movida
para qualquer lugar. Ele se atualiza devagar, mas só onde não há pessoa, para
acompanhar mudanças de luz sem "aprender" quem está na frente.
"""
import cv2
import numpy as np

from . import config


class BackgroundModel:
    def __init__(self):
        self._open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.MORPH_OPEN_KERNEL, config.MORPH_OPEN_KERNEL))
        self._close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.MORPH_CLOSE_KERNEL, config.MORPH_CLOSE_KERNEL))
        self.reset()

    def reset(self):
        self._acc = None
        self._count = 0
        self._bg = None  # float32, mesma forma do frame

    @property
    def ready(self):
        return self._bg is not None

    @property
    def progress(self):
        return min(self._count / config.CALIBRATION_FRAMES, 1.0)

    def add_calibration_frame(self, frame):
        """Média de vários frames da cena vazia: reduz o ruído da câmera."""
        f = frame.astype(np.float32)
        self._acc = f if self._acc is None else self._acc + f
        self._count += 1
        if self._count >= config.CALIBRATION_FRAMES:
            self._bg = self._acc / self._count

    def image(self):
        return cv2.convertScaleAbs(self._bg)

    def person_mask(self, frame):
        """Máscara binária (0/255) de tudo que difere do fundo."""
        diff = cv2.absdiff(frame, self.image())          # Lab07 (Desafio 1)
        diff = diff.max(axis=2)                          # maior diferença entre B, G e R
        diff = cv2.GaussianBlur(diff, (5, 5), 0)         # Lab04: reduz ruído do sensor
        _, mask = cv2.threshold(diff, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)    # Lab06: tira pontinhos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)  # Lab06: fecha buracos
        if config.MASK_DILATE:
            mask = cv2.dilate(mask, self._open_k, iterations=config.MASK_DILATE)
        return mask

    def update(self, frame, person_mask):
        """Aprende o fundo devagar, apenas fora da pessoa."""
        background_only = cv2.bitwise_not(person_mask)
        cv2.accumulateWeighted(frame, self._bg, config.BG_LEARNING_RATE, mask=background_only)
