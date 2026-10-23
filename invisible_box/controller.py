"""Máquina de estados que transforma gestos em ações sobre a janela."""
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import config, gestures
from .box import InvisibleBox


class State(Enum):
    DEFINING = "DEFININDO"
    FREE = "LIVRE"
    DRAGGING = "ARRASTANDO"
    RESIZING = "REDIMENSIONANDO"
    TRANSFORMING = "GIRANDO/ESCALANDO"
    INERTIA = "DESLIZANDO"


HINTS = {
    State.DEFINING: "Aponte os DOIS indicadores nos cantos e faca uma pinca rapida para salvar",
    State.FREE: "Pinca dentro = mover | canto = redimensionar | 2 maos = girar | 2 maos abertas = recriar",
    State.DRAGGING: "Solte rapido para arremessar",
    State.RESIZING: "Solte para fixar o tamanho",
    State.TRANSFORMING: "Gire e afaste as maos",
    State.INERTIA: "Pinca na janela para pegar de novo",
}


@dataclass
class HandInfo:
    key: str
    points: np.ndarray
    pinch: gestures.PinchEvent
    point: np.ndarray       # ponto da pinça suavizado
    tip: np.ndarray         # ponta do indicador suavizada
    open_palm: bool


class BoxController:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.state = State.DEFINING
        self.box = None
        self.preview = None          # (p1, p2) enquanto define a janela
        self.hands = []
        self._pinch = gestures.PinchDetector()
        self._smooth = {}
        self._last_pointing = {}     # chave da mão -> último instante com o indicador esticado
        self._corners = deque()      # (t, ponta A, ponta B) dos últimos instantes
        self._tap = None             # pinça rápida em andamento enquanto define a janela
        self._t = 0.0
        self._reset_frames = 0
        self._wait_release = False
        self._grab = {}

    # --- API ---------------------------------------------------------------
    def new_box(self):
        self.box = None
        self._grab = {}
        self._tap = None
        self._corners.clear()
        self.state = State.DEFINING

    @property
    def hint(self):
        return HINTS[self.state]

    def update(self, hands, t, dt):
        self._t = t
        self.hands = self._read_hands(hands, t)
        if self.box is not None:
            self.box.wobble *= 0.9
        handler = {
            State.DEFINING: self._defining,
            State.FREE: self._free,
            State.DRAGGING: self._dragging,
            State.RESIZING: self._resizing,
            State.TRANSFORMING: self._transforming,
            State.INERTIA: self._inertia,
        }[self.state]
        handler(dt)

    # --- leitura das mãos ---------------------------------------------------
    def _read_hands(self, hands, t):
        infos, seen = [], set()
        for i, h in enumerate(hands):
            key = h.label if h.label not in seen else f"{h.label}{i}"
            seen.add(key)
            pts = h.points
            if gestures.finger_extended(pts, "index"):
                self._last_pointing[key] = t
            # pinça a partir do "apontar" (outros dedos dobrados) também vale:
            # o indicador estava esticado há pouco, então não é um punho.
            pointed = t - self._last_pointing.get(key, -math.inf) <= config.POINTING_PINCH_WINDOW
            allowed = gestures.pinch_allowed(pts) or pointed
            ev = self._pinch.update(key, gestures.pinch_ratio(pts), allowed)
            p = self._smoothed(key, gestures.pinch_point(pts))
            tip = self._smoothed(key + "/tip", pts[gestures.INDEX_TIP])
            infos.append(HandInfo(key, pts, ev, p, tip, gestures.is_open_palm(pts)))
        self._pinch.forget_missing(seen)  # mão que sumiu conta como pinça solta
        alive = seen | {k + "/tip" for k in seen}
        self._smooth = {k: v for k, v in self._smooth.items() if k in alive}
        self._last_pointing = {k: v for k, v in self._last_pointing.items() if k in seen}
        return infos

    def _smoothed(self, key, raw):
        """Média móvel exponencial da posição, para o ponto não tremer."""
        raw = np.asarray(raw, np.float32)
        prev = self._smooth.get(key)
        a = config.POINT_SMOOTHING
        p = raw if prev is None else prev * a + raw * (1 - a)
        self._smooth[key] = p
        return p

    def _hand(self, key):
        return next((h for h in self.hands if h.key == key), None)

    def _pinching(self):
        return [h for h in self.hands if h.pinch.pinching]

    # --- estados ------------------------------------------------------------
    def _defining(self, dt):
        """Os dois indicadores marcam cantos opostos; uma pinça rápida salva a janela."""
        t = self._t
        if len(self.hands) < 2:
            self.preview, self._tap = None, None
            self._corners.clear()
            return
        a, b = self.hands[0], self.hands[1]
        self._corners.append((t, a.tip.copy(), b.tip.copy()))
        while self._corners and t - self._corners[0][0] > 1.0:
            self._corners.popleft()

        if self._tap is None:
            started = next((h for h in (a, b) if h.pinch.started), None)
            if started is not None:
                # durante a pinça o indicador desce até o polegar e o canto "anda";
                # por isso guardamos os cantos de um instante ANTES da pinça.
                self._tap = {"key": started.key, "t0": t, "corners": self._corners_before(t)}

        if self._tap is not None:
            h = self._hand(self._tap["key"])
            if h is None or t - self._tap["t0"] > config.TAP_MAX_SECONDS:
                self._tap = None                      # segurou demais: não é pinça rápida
            elif h.pinch.released:                    # fechou e abriu rápido -> salva (Lab12)
                p1, p2 = self._tap["corners"]
                self.box = InvisibleBox.from_corners(p1, p2)
                self.preview, self._tap = None, None
                self._corners.clear()
                self._wait_release = True             # só aceita novos gestos depois de soltar
                self.state = State.FREE
                return

        self.preview = self._tap["corners"] if self._tap else (a.tip.copy(), b.tip.copy())

    def _corners_before(self, t):
        """Cantos de TAP_LOOKBACK segundos atrás (ou os mais antigos que houver)."""
        target = t - config.TAP_LOOKBACK
        chosen = self._corners[0]
        for item in self._corners:
            if item[0] > target:
                break
            chosen = item
        return chosen[1].copy(), chosen[2].copy()

    @property
    def tap_pending(self):
        return self._tap is not None

    def _check_reset(self):
        both_open = len(self.hands) >= 2 and all(h.open_palm for h in self.hands[:2])
        self._reset_frames = self._reset_frames + 1 if both_open else 0
        if self._reset_frames >= config.RESET_CONFIRM_FRAMES:
            self._reset_frames = 0
            self.new_box()
            return True
        return False

    @property
    def reset_progress(self):
        return self._reset_frames / config.RESET_CONFIRM_FRAMES

    def _try_grab(self):
        """Começa uma interação se alguma pinça acabou de fechar sobre a janela."""
        box = self.box
        pinching = [h for h in self._pinching() if box.contains(h.point, config.GRAB_MARGIN * 3)]
        started = [h for h in pinching if h.pinch.started]
        if not started:
            return False

        if len(pinching) >= 2:
            self._start_transform(pinching[0], pinching[1])
            return True

        h = started[0]
        idx, d = box.nearest_corner(h.point)
        if d <= config.CORNER_GRAB_RADIUS:
            self._grab = {"key": h.key, "corner": idx}
            self.state = State.RESIZING
            return True
        if box.contains(h.point, config.GRAB_MARGIN):
            self._start_drag(h)
            return True
        return False

    def _start_drag(self, h):
        self.box.velocity[:] = 0
        self._grab = {"key": h.key, "offset": self.box.center - h.point, "vel": np.zeros(2, np.float32)}
        self.state = State.DRAGGING

    def _start_transform(self, a, b):
        v = b.point - a.point
        self._grab = {
            "keys": (a.key, b.key),
            "angle0": math.atan2(v[1], v[0]),
            "dist0": max(float(np.linalg.norm(v)), 1.0),
            "mid0": (a.point + b.point) / 2,
            "box0": (self.box.center.copy(), self.box.w, self.box.h, self.box.angle),
        }
        self.box.velocity[:] = 0
        self.state = State.TRANSFORMING

    def _free(self, dt):
        if self.box is None:
            self.state = State.DEFINING
            return
        if self._check_reset():
            return
        if self._wait_release:
            if not self._pinching():
                self._wait_release = False
            return
        self._try_grab()

    def _dragging(self, dt):
        g = self._grab
        h = self._hand(g["key"])
        if h is None or not h.pinch.pinching:
            speed = float(np.linalg.norm(g["vel"]))
            if speed >= config.THROW_MIN_SPEED:
                self.box.velocity = g["vel"].copy()
                self.state = State.INERTIA
            else:
                self.state = State.FREE
            return

        # segunda mão fez pinça na janela -> girar/escalar
        other = [o for o in self._pinching()
                 if o.key != h.key and o.pinch.started and self.box.contains(o.point, config.GRAB_MARGIN * 3)]
        if other:
            self._start_transform(h, other[0])
            return

        new_center = h.point + g["offset"]
        if dt > 0:
            inst = (new_center - self.box.center) / dt
            g["vel"] = g["vel"] * 0.6 + inst * 0.4
        self.box.center = new_center.astype(np.float32)
        self.box.keep_inside(self.width, self.height)
        self.box.wobble = max(self.box.wobble, min(float(np.linalg.norm(g["vel"])) / 900.0, 1.0))

    def _resizing(self, dt):
        h = self._hand(self._grab["key"])
        if h is None or not h.pinch.pinching:
            self.state = State.FREE
            return
        self.box.resize_with_corner(self._grab["corner"], h.point)
        self.box.keep_inside(self.width, self.height)

    def _transforming(self, dt):
        g = self._grab
        a, b = self._hand(g["keys"][0]), self._hand(g["keys"][1])
        a_ok = a is not None and a.pinch.pinching
        b_ok = b is not None and b.pinch.pinching
        if not (a_ok and b_ok):
            remaining = a if a_ok else b if b_ok else None
            if remaining is not None:
                self._start_drag(remaining)   # continua arrastando com a mão que ficou
            else:
                self.state = State.FREE
            return

        v = b.point - a.point
        c0, w0, h0, ang0 = g["box0"]
        scale = float(np.linalg.norm(v)) / g["dist0"]
        self.box.angle = ang0 + (math.atan2(v[1], v[0]) - g["angle0"])
        self.box.w = max(w0 * scale, config.MIN_BOX_SIZE)
        self.box.h = max(h0 * scale, config.MIN_BOX_SIZE)
        self.box.center = (c0 + ((a.point + b.point) / 2 - g["mid0"])).astype(np.float32)
        self.box.keep_inside(self.width, self.height)

    def _inertia(self, dt):
        if self._try_grab():
            return
        if not self.box.step_physics(dt, self.width, self.height):
            self.state = State.FREE
