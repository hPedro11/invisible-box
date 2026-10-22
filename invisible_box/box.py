"""A janela de invisibilidade: um retângulo que pode mover, girar, mudar de tamanho e deslizar."""
import math

import cv2
import numpy as np

from . import config

# cantos em coordenadas locais, em sentido horário: 0=sup-esq, 1=sup-dir, 2=inf-dir, 3=inf-esq
_LOCAL_SIGNS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)


class InvisibleBox:
    def __init__(self, cx, cy, w, h, angle=0.0):
        self.center = np.array([cx, cy], dtype=np.float32)
        self.w, self.h = float(w), float(h)
        self.angle = float(angle)          # radianos
        self.velocity = np.zeros(2, dtype=np.float32)  # px/s
        self.wobble = 0.0                  # 0..1, ondulação das bordas

    @classmethod
    def from_corners(cls, p1, p2):
        p1, p2 = np.asarray(p1, np.float32), np.asarray(p2, np.float32)
        c = (p1 + p2) / 2
        w = max(abs(p2[0] - p1[0]), config.MIN_BOX_SIZE)
        h = max(abs(p2[1] - p1[1]), config.MIN_BOX_SIZE)
        return cls(c[0], c[1], w, h)

    # --- geometria ------------------------------------------------------------
    def axes(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([c, s], np.float32), np.array([-s, c], np.float32)

    def corners(self):
        ux, uy = self.axes()
        half = _LOCAL_SIGNS * np.array([self.w / 2, self.h / 2], np.float32)
        return self.center + half[:, :1] * ux + half[:, 1:] * uy

    def to_local(self, pt):
        ux, uy = self.axes()
        d = np.asarray(pt, np.float32) - self.center
        return np.array([d @ ux, d @ uy], np.float32)

    def contains(self, pt, margin=0.0):
        lx, ly = self.to_local(pt)
        return abs(lx) <= self.w / 2 + margin and abs(ly) <= self.h / 2 + margin

    def nearest_corner(self, pt):
        d = np.linalg.norm(self.corners() - np.asarray(pt, np.float32), axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def bbox(self):
        return cv2.boundingRect(np.round(self.polygon()).astype(np.int32))

    def polygon(self, t=0.0):
        """Contorno da janela. Com wobble > 0, as bordas ondulam como gelatina."""
        corners = self.corners()
        if self.wobble < 0.02:
            return corners
        pts = []
        n = 24
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            edge = b - a
            length = float(np.linalg.norm(edge)) or 1.0
            normal = np.array([edge[1], -edge[0]], np.float32) / length
            for j in range(n):
                s = j / n
                bump = math.sin(math.pi * s)  # zero nos cantos
                off = math.sin(s * 2 * math.pi * 2 + t * 12 + i) * bump
                pts.append(a + edge * s + normal * off * config.WOBBLE_AMPLITUDE * self.wobble)
        return np.array(pts, np.float32)

    def mask(self, shape, t=0.0):
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(self.polygon(t)).astype(np.int32)], 255)   # Lab01/Lab05
        return m

    # --- manipulação --------------------------------------------------------
    def resize_with_corner(self, idx, pt):
        """Arrasta o canto idx; o canto oposto fica parado (funciona com a janela girada)."""
        ux, uy = self.axes()
        opposite = self.corners()[(idx + 2) % 4]
        d = np.asarray(pt, np.float32) - opposite
        dx, dy = float(d @ ux), float(d @ uy)
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        self.w = max(abs(dx), config.MIN_BOX_SIZE)
        self.h = max(abs(dy), config.MIN_BOX_SIZE)
        self.center = opposite + ux * (sx * self.w / 2) + uy * (sy * self.h / 2)

    def keep_inside(self, width, height):
        """Garante que o centro não saia da tela (a janela pode ficar parcialmente fora)."""
        self.center[0] = np.clip(self.center[0], 0, width - 1)
        self.center[1] = np.clip(self.center[1], 0, height - 1)

    def step_physics(self, dt, width, height):
        """Inércia + atrito + quique nas bordas. Retorna True enquanto ainda se move."""
        self.center += self.velocity * dt
        self.velocity *= config.FRICTION ** (dt * 60.0)

        x, y, bw, bh = self.bbox()
        if x < 0 and self.velocity[0] < 0 or x + bw > width and self.velocity[0] > 0:
            self.velocity[0] *= -config.BOUNCE
            self.center[0] += -x if x < 0 else width - (x + bw)
        if y < 0 and self.velocity[1] < 0 or y + bh > height and self.velocity[1] > 0:
            self.velocity[1] *= -config.BOUNCE
            self.center[1] += -y if y < 0 else height - (y + bh)

        speed = float(np.linalg.norm(self.velocity))
        self.wobble = max(self.wobble * 0.95, min(speed / 900.0, 1.0))
        if speed < config.STOP_SPEED:
            self.velocity[:] = 0
            return False
        return True
