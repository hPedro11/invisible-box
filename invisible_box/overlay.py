"""Desenhos na tela: contorno da janela, mãos, HUD e tela de calibração (Lab01)."""
import cv2
import numpy as np

from .controller import State

CYAN = (255, 220, 80)
WHITE = (255, 255, 255)
GREEN = (120, 255, 120)
ORANGE = (40, 170, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_box(frame, controller, t):
    box = controller.box
    if box is None:
        return
    active = controller.state in (State.DRAGGING, State.RESIZING, State.TRANSFORMING)
    color = ORANGE if active else CYAN
    poly = np.round(box.polygon(t)).astype(np.int32)
    # brilho: linha grossa borrada por baixo + linha fina por cima
    glow = np.zeros_like(frame)
    cv2.polylines(glow, [poly], True, color, 6, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 6)
    cv2.add(frame, glow, dst=frame)
    cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)

    if controller.state in (State.FREE, State.RESIZING):
        for i, c in enumerate(box.corners()):
            on = controller.state == State.RESIZING and controller._grab.get("corner") == i
            cv2.circle(frame, _pt(c), 7 if on else 4, ORANGE if on else color, -1, cv2.LINE_AA)


def draw_preview(frame, controller):
    if controller.preview is None:
        return
    p1, p2 = controller.preview
    cv2.rectangle(frame, _pt(p1), _pt(p2), WHITE, 1, cv2.LINE_AA)
    prog = controller.define_progress
    if prog > 0:
        x, y = _pt(np.minimum(p1, p2))
        cv2.rectangle(frame, (x, y - 10), (x + int(120 * prog), y - 4), GREEN, -1)


def draw_hands(frame, controller, show_skeleton):
    for h in controller.hands:
        if show_skeleton:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, _pt(h.points[a]), _pt(h.points[b]), (200, 200, 200), 1, cv2.LINE_AA)
        color = GREEN if h.pinch.pinching else WHITE
        cv2.circle(frame, _pt(h.point), 9 if h.pinch.pinching else 5, color,
                   -1 if h.pinch.pinching else 2, cv2.LINE_AA)
    prog = controller.reset_progress
    if prog > 0:
        cv2.putText(frame, f"Recriando janela... {int(prog * 100)}%", (20, 90),
                    FONT, 0.6, ORANGE, 2, cv2.LINE_AA)


def draw_hud(frame, controller, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{controller.state.value}", (10, 23), FONT, 0.6, CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.0f} FPS", (w - 95, 23), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, controller.hint, (10, h - 10), FONT, 0.48, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "[N] nova  [R] recalibrar  [D] debug  [H] maos  [Q] sair",
                (w - 420, 23), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def draw_calibration(frame, progress):
    h, w = frame.shape[:2]
    shade = (frame * 0.4).astype(np.uint8)
    frame[:] = shade
    msg = "Saia da frente da camera"
    (tw, _), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
    cv2.putText(frame, msg, ((w - tw) // 2, h // 2 - 20), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
    bar_w = 360
    x0 = (w - bar_w) // 2
    cv2.rectangle(frame, (x0, h // 2 + 10), (x0 + bar_w, h // 2 + 26), (90, 90, 90), 1)
    cv2.rectangle(frame, (x0, h // 2 + 10), (x0 + int(bar_w * progress), h // 2 + 26), CYAN, -1)


def debug_panel(frame, background, person_mask):
    """Miniaturas para a apresentação: fundo aprendido e máscara da pessoa."""
    h, w = frame.shape[:2]
    tw, th = w // 4, h // 4
    bg_small = cv2.resize(background, (tw, th))
    mask_small = cv2.cvtColor(cv2.resize(person_mask, (tw, th)), cv2.COLOR_GRAY2BGR)
    y = 40
    frame[y:y + th, w - tw - 10:w - 10] = bg_small
    frame[y + th + 6:y + 2 * th + 6, w - tw - 10:w - 10] = mask_small
    for label, ly in (("fundo", y), ("mascara", y + th + 6)):
        cv2.rectangle(frame, (w - tw - 10, ly), (w - tw + 62, ly + 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (w - tw - 5, ly + 15), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
