"""Desenhos na tela: contorno da janela, mãos, HUD e painel de debug (Lab01)."""
import cv2
import numpy as np

from . import config
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


def _thin_outline(frame, poly, color):
    """Contorno fino e translúcido (linha desenhada numa cópia e misturada)."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(poly)
    m = config.BOX_LINE_WIDTH + 2
    x0, y0 = max(bx - m, 0), max(by - m, 0)
    x1, y1 = min(bx + bw + m, w), min(by + bh + m, h)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    layer = region.copy()
    cv2.polylines(layer, [poly - (x0, y0)], True, color, config.BOX_LINE_WIDTH, cv2.LINE_AA)
    a = config.BOX_LINE_ALPHA
    frame[y0:y1, x0:x1] = cv2.addWeighted(layer, a, region, 1.0 - a, 0.0)


def draw_box(frame, controller, t):
    """A janela: só uma linha fina, como no vídeo de referência.

    Os cantos ficam escondidos e aparecem quando uma mão chega perto deles
    (é quando dá para redimensionar).
    """
    box = controller.box
    if box is None:
        return
    active = controller.state in (State.DRAGGING, State.RESIZING, State.TRANSFORMING)
    color = config.BOX_ACTIVE_COLOR if active else config.BOX_COLOR
    poly = np.round(box.polygon(t)).astype(np.int32)
    _thin_outline(frame, poly, color)

    if controller.state not in (State.FREE, State.RESIZING):
        return
    grabbed = controller._grab.get("corner") if controller.state == State.RESIZING else None
    for i, c in enumerate(box.corners()):
        if i == grabbed:
            cv2.circle(frame, _pt(c), 5, config.BOX_ACTIVE_COLOR, -1, cv2.LINE_AA)
        elif any(np.linalg.norm(h.point - c) < config.BOX_CORNER_HINT for h in controller.hands):
            cv2.circle(frame, _pt(c), 5, color, 1, cv2.LINE_AA)


def draw_missing_notice(frame, box, missing):
    """Aviso dentro da janela: parte do fundo dela nunca foi vista (fica borrada).

    Aparece no topo da janela, onde é impossível não ver, com a porcentagem.
    """
    lines = [f"{missing * 100:.0f}% do fundo desta janela nunca foi visto (fica borrado)",
             "saia do quadro por 2 s para a camera ver o que esta atras de voce"]
    x, y, bw, bh = box.bbox()
    h, w = frame.shape[:2]
    sizes = [cv2.getTextSize(t, FONT, 0.55, 1)[0] for t in lines]
    tw = max(sz[0] for sz in sizes) + 20
    th = sum(sz[1] for sz in sizes) + 30
    x0 = int(min(max(x + (bw - tw) // 2, 0), max(w - tw, 0)))
    y0 = int(min(max(y + 12, 40), max(h - th - 40, 40)))
    region = frame[y0:y0 + th, x0:x0 + tw]
    frame[y0:y0 + th, x0:x0 + tw] = cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.65, 0.0)
    ty = y0 + 8
    for text, (_, sh) in zip(lines, sizes):
        ty += sh + 6
        cv2.putText(frame, text, (x0 + 10, ty), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)


def draw_preview(frame, controller):
    """Retângulo entre os dois indicadores, no mesmo estilo da janela."""
    if controller.preview is None:
        return
    p1, p2 = controller.preview
    x0, y0 = _pt(np.minimum(p1, p2))
    x1, y1 = _pt(np.maximum(p1, p2))
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
    if controller.tap_pending:
        _thin_outline(frame, poly, config.BOX_ACTIVE_COLOR)
        cv2.putText(frame, "solte a pinca para salvar", (x0, y0 - 8), FONT, 0.5,
                    config.BOX_ACTIVE_COLOR, 1, cv2.LINE_AA)
    else:
        _thin_outline(frame, poly, config.BOX_COLOR)


def draw_hands(frame, controller, show_skeleton):
    for h in controller.hands:
        if show_skeleton:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, _pt(h.points[a]), _pt(h.points[b]), (200, 200, 200), 1, cv2.LINE_AA)
        color = GREEN if h.pinch.pinching else WHITE
        # definindo a janela, o que importa é a ponta do indicador; depois, o ponto da pinça
        p = h.tip if controller.state == State.DEFINING else h.point
        cv2.circle(frame, _pt(p), 9 if h.pinch.pinching else 5, color,
                   -1 if h.pinch.pinching else 2, cv2.LINE_AA)
    prog = controller.reset_progress
    if prog > 0:
        cv2.putText(frame, f"Recriando janela... {int(prog * 100)}%", (20, 90),
                    FONT, 0.6, ORANGE, 2, cv2.LINE_AA)


def draw_hud(frame, controller, fps, known=None):
    """Barra de cima: estado, quanto do fundo já foi visto e FPS. Barra de baixo: dica."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{controller.state.value}", (10, 23), FONT, 0.6, CYAN, 2, cv2.LINE_AA)
    if known is not None:
        cv2.putText(frame, f"fundo {known * 100:3.0f}%", (215, 23), FONT, 0.5,
                    GREEN if known > 0.97 else ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.0f} FPS", (w - 95, 23), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, controller.hint, (10, h - 10), FONT, 0.48, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "[N] nova  [R] reaprender fundo  [D] debug  [H] maos  [Q] sair",
                (w - 470, 23), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def debug_panel(frame, background, person_mask, unknown=None):
    """Miniaturas para a apresentação: fundo aprendido e máscara da pessoa.

    Com `unknown`, as áreas de fundo nunca vistas (só "chutadas" pelo inpaint)
    aparecem em magenta.
    """
    h, w = frame.shape[:2]
    tw, th = w // 4, h // 4
    bg_small = cv2.resize(background, (tw, th))
    if unknown is not None:
        miss = cv2.resize(unknown, (tw, th), interpolation=cv2.INTER_NEAREST) > 0
        bg_small[miss] = (bg_small[miss] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
    mask_small = cv2.cvtColor(cv2.resize(person_mask, (tw, th)), cv2.COLOR_GRAY2BGR)
    y = 40
    frame[y:y + th, w - tw - 10:w - 10] = bg_small
    frame[y + th + 6:y + 2 * th + 6, w - tw - 10:w - 10] = mask_small
    for label, ly in (("fundo aprendido", y), ("pessoas", y + th + 6)):
        cv2.rectangle(frame, (w - tw - 10, ly), (w - tw + 120, ly + 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (w - tw - 5, ly + 15), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
