"""Reconhecimento de gestos a partir dos 21 landmarks da mão (Lab13).

Tudo é calculado com distâncias normalizadas pelo tamanho da mão, então os
gestos funcionam igual perto ou longe da câmera.
"""
from dataclasses import dataclass

import numpy as np

from . import config

WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

FINGERS = {  # nome: (ponta, articulação PIP)
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def hand_size(pts):
    """Distância pulso -> base do dedo médio: a 'régua' da mão."""
    return max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)


def pinch_ratio(pts):
    return _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size(pts)


def pinch_point(pts):
    return (np.asarray(pts[THUMB_TIP]) + np.asarray(pts[INDEX_TIP])) / 2.0


def finger_extended(pts, name):
    """Dedo esticado = ponta mais longe do pulso que a articulação.

    Diferente do teste 'ponta acima da articulação', funciona com a mão girada.
    """
    tip, pip = FINGERS[name]
    return _dist(pts[WRIST], pts[tip]) > _dist(pts[WRIST], pts[pip]) * 1.1


def extended_fingers(pts):
    return {name: finger_extended(pts, name) for name in FINGERS}


def is_fist(pts):
    return not any(extended_fingers(pts).values())


def is_open_palm(pts):
    return all(extended_fingers(pts).values()) and pinch_ratio(pts) > 0.6


def pinch_allowed(pts):
    """Evita confundir punho com pinça: algum dos 3 últimos dedos precisa estar esticado."""
    if not config.PINCH_REQUIRE_OPEN_FINGERS:
        return True
    ext = extended_fingers(pts)
    return ext["middle"] or ext["ring"] or ext["pinky"]


@dataclass
class PinchEvent:
    pinching: bool
    started: bool
    released: bool


class PinchDetector:
    """Estado da pinça por mão, com histerese (dois limiares) e eventos de borda."""

    def __init__(self, on=config.PINCH_ON, off=config.PINCH_OFF):
        self.on, self.off = on, off
        self._state = {}

    def update(self, key, ratio, allowed=True):
        was = self._state.get(key, False)
        if was:
            now = ratio < self.off
        else:
            now = allowed and ratio < self.on
        self._state[key] = now
        return PinchEvent(now, now and not was, was and not now)

    def forget_missing(self, present_keys):
        """Mãos que sumiram da imagem contam como pinça solta. Retorna as chaves soltas."""
        released = [k for k, v in self._state.items() if k not in present_keys and v]
        for k in list(self._state):
            if k not in present_keys:
                del self._state[k]
        return released
