"""Detecção das mãos com o MediaPipe HandLandmarker, API Tasks em modo VIDEO (Lab13)."""
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from . import config

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class Hand:
    label: str          # "Left" / "Right" (do ponto de vista do usuário, com imagem espelhada)
    points: np.ndarray  # (21, 2) em pixels
    score: float


class HandTracker:
    def __init__(self, model_path=config.MODEL_PATH):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ts = -1

    def detect(self, frame_bgr):
        """Recebe o frame ORIGINAL (BGR) e devolve a lista de mãos em pixels."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # o modelo espera RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._last_ts:  # o modo VIDEO exige timestamps crescentes
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(image, ts)
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            label, score = f"Hand{i}", 1.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                label, score = cat.category_name, cat.score
            hands.append(Hand(label, pts, score))
        return hands

    def close(self):
        self._landmarker.close()
