"""Segmentação de pessoas com o MediaPipe ImageSegmenter, API Tasks em modo VIDEO.

Substitui a subtração de fundo na hora de achar QUEM está na cena: o modelo
reconhece pessoas mesmo paradas e mesmo que já estivessem lá desde o início.
"""
import time

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .imaging import reduced_copy

_vision = mp.tasks.vision


class PersonSegmenter:
    def __init__(self, model_path=config.SEGMENTER_PATH):
        options = _vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=_vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = _vision.ImageSegmenter.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ts = -1
        self._prob = None

    def probability(self, frame_bgr):
        """Probabilidade (0..1, float32) de cada pixel ser pessoa, suavizada no tempo.

        Roda numa cópia reduzida (PROCESS_WIDTH), então o resultado também vem
        reduzido; BackgroundModel.person_mask leva de volta ao tamanho do frame.
        """
        rgb = cv2.cvtColor(reduced_copy(frame_bgr), cv2.COLOR_BGR2RGB)  # o modelo espera RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._last_ts:  # o modo VIDEO exige timestamps crescentes
            ts = self._last_ts + 1
        self._last_ts = ts

        masks = self._segmenter.segment_for_video(image, ts).confidence_masks
        first = np.squeeze(masks[0].numpy_view())
        # selfie_segmenter: uma máscara só, já é "pessoa".
        # multiclasse: a máscara 0 é o fundo e as outras são partes da pessoa.
        prob = first if len(masks) == 1 else 1.0 - first
        prob = prob.astype(np.float32)

        a = config.SEG_SMOOTHING
        if self._prob is None or self._prob.shape != prob.shape or a <= 0:
            self._prob = prob.copy()
        else:
            self._prob = self._prob * a + prob * (1 - a)
        return self._prob

    def close(self):
        self._segmenter.close()
