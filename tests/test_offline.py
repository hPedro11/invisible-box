"""Testes sem webcam: mãos sintéticas e frames sintéticos.

Rodar a partir da raiz do projeto:  python -m tests.test_offline
"""
import math
from pathlib import Path

import cv2
import numpy as np

from invisible_box import config, gestures
from invisible_box.background import BackgroundModel
from invisible_box.box import InvisibleBox
from invisible_box.controller import BoxController, State
from invisible_box.glass import GlassEffect
from invisible_box.hand_tracker import Hand
from invisible_box.person_segmenter import PersonSegmenter

# --- mão sintética (21 landmarks, unidades de "tamanho da mão") -------------
OPEN = np.array([
    [0, 0],
    [-0.3, -0.2], [-0.5, -0.45], [-0.65, -0.7], [-0.8, -0.9],        # polegar
    [-0.25, -1.0], [-0.3, -1.4], [-0.32, -1.65], [-0.33, -1.9],      # indicador
    [0, -1.0], [0, -1.5], [0, -1.8], [0, -2.1],                      # médio
    [0.25, -0.95], [0.3, -1.4], [0.32, -1.65], [0.33, -1.9],         # anelar
    [0.45, -0.85], [0.55, -1.2], [0.58, -1.4], [0.6, -1.6],          # mínimo
], dtype=np.float32)


def make_hand(target, pinch=False, fist=False, point=False, label="Right", scale=80.0):
    """Mão sintética. Com point=True o indicador aponta (outros dedos dobrados) e
    `target` é a ponta do indicador; senão `target` é o ponto da pinça."""
    pts = OPEN.copy()
    curled = [(5, 8), (9, 12), (13, 16), (17, 20)] if fist else [(9, 12), (13, 16), (17, 20)] if point else []
    for mcp, tip in curled:
        pts[tip] = pts[mcp] * 0.8           # ponta encolhida perto da palma
        pts[tip - 1] = pts[mcp] * 0.85
    if fist:
        pts[4] = pts[5] * 0.9
    if pinch:
        pts[4] = pts[8] + np.array([0.05, 0.0])
    pts *= scale
    anchor = pts[8] if point else (pts[4] + pts[8]) / 2
    pts += np.asarray(target, np.float32) - anchor
    return Hand(label, pts, 1.0)


def test_gestures():
    open_h = make_hand((300, 300)).points
    pinch_h = make_hand((300, 300), pinch=True).points
    fist_h = make_hand((300, 300), fist=True).points
    assert gestures.pinch_ratio(open_h) > config.PINCH_OFF
    assert gestures.pinch_ratio(pinch_h) < config.PINCH_ON
    assert gestures.is_open_palm(open_h)
    assert gestures.is_fist(fist_h) and not gestures.is_fist(open_h)
    assert gestures.pinch_allowed(pinch_h) and not gestures.pinch_allowed(fist_h)

    det = gestures.PinchDetector()
    assert det.update("R", 0.2).started
    assert det.update("R", 0.38).pinching          # entre os limiares: continua fechada
    assert det.update("R", 0.6).released
    print("gestos: ok")


_clock = [0.0]


def run(ctrl, hands, frames=1, dt=1 / 30):
    for _ in range(frames):
        _clock[0] += dt
        ctrl.update(hands, _clock[0], dt)


def test_controller():
    W, H = 960, 540
    c = BoxController(W, H)
    assert c.state == State.DEFINING

    # os dois indicadores apontam para os cantos: só mostra a prévia
    a, b = (250, 150), (550, 400)
    pointing = [make_hand(a, point=True, label="Left"), make_hand(b, point=True)]
    run(c, pointing, 10)
    assert c.state == State.DEFINING and c.preview is not None

    # pinça SEGURADA (lenta) não salva
    run(c, [make_hand(a, pinch=True, point=True, label="Left"), make_hand(b, point=True)], 30)
    run(c, pointing, 10)
    assert c.state == State.DEFINING and c.box is None, c.state

    # pinça rápida salva; o dedo "anda" durante a pinça, mas valem os cantos de antes
    moved = (a[0] + 25, a[1] + 30)
    run(c, [make_hand(moved, pinch=True, point=True, label="Left"), make_hand(b, point=True)], 5)
    assert c.tap_pending
    run(c, pointing, 1)
    assert c.state == State.FREE and c.box is not None, c.state
    assert abs(c.box.w - 300) < 3 and abs(c.box.h - 250) < 3, (c.box.w, c.box.h)
    print(f"definir com pinça rápida: ok, janela {c.box.w:.0f} x {c.box.h:.0f}")
    run(c, [], 2)                                   # mãos saem

    # arrastar pelo meio e soltar rápido -> desliza
    run(c, [make_hand((400, 275))], 2)
    run(c, [make_hand((400, 275), True)])
    assert c.state == State.DRAGGING, c.state
    for i in range(1, 6):
        run(c, [make_hand((400 + 40 * i, 275), True)])
    moved_to = c.box.center.copy()
    assert moved_to[0] > 480
    run(c, [make_hand((640, 275))])
    assert c.state == State.INERTIA, c.state
    run(c, [], 200)
    assert c.state == State.FREE and c.box.center[0] > moved_to[0]
    print("arrastar + inércia: ok, centro final", np.round(c.box.center))

    # redimensionar pelo canto inferior direito
    corner = c.box.corners()[2]
    run(c, [make_hand(corner)], 2)
    run(c, [make_hand(corner, True)])
    assert c.state == State.RESIZING, c.state
    old_w = c.box.w
    run(c, [make_hand(corner - np.array([60, 0]), True)], 3)
    run(c, [], 1)
    assert c.state == State.FREE and c.box.w < old_w
    print(f"redimensionar: ok, largura {old_w:.0f} -> {c.box.w:.0f}")

    # girar com duas mãos
    cx, cy = c.box.center
    a0, b0 = (cx - 50, cy), (cx + 50, cy)
    run(c, [make_hand(a0, label="Left"), make_hand(b0)], 2)
    run(c, [make_hand(a0, True, label="Left"), make_hand(b0, True)])
    assert c.state == State.TRANSFORMING, c.state
    ang = math.radians(30)
    a1 = (cx - 50 * math.cos(ang), cy - 50 * math.sin(ang))
    b1 = (cx + 50 * math.cos(ang), cy + 50 * math.sin(ang))
    for _ in range(6):   # repetido para a suavização convergir
        run(c, [make_hand(a1, True, label="Left"), make_hand(b1, True)])
    assert abs(math.degrees(c.box.angle) - 30) < 3, math.degrees(c.box.angle)
    run(c, [], 1)
    assert c.state == State.FREE
    print(f"girar: ok, ângulo {math.degrees(c.box.angle):.1f} graus")

    # recriar: duas mãos abertas
    open2 = [make_hand((200, 300), label="Left"), make_hand((700, 300))]
    run(c, open2, config.RESET_CONFIRM_FRAMES)
    assert c.state == State.DEFINING and c.box is None
    print("recriar janela: ok")


def synthetic_scene(w=960, h=540):
    """Fundo com textura (prédios e listras) para a refração ficar visível."""
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        bg[y] = (235 - y * 0.15, 200 - y * 0.1, 140)   # céu
    cv2.rectangle(bg, (0, 330), (w, h), (90, 110, 100), -1)
    for i, x in enumerate(range(40, w, 110)):
        top = 120 + (i * 53) % 140
        cv2.rectangle(bg, (x, top), (x + 80, 380), (120 + i * 9 % 80, 120, 150), -1)
        for wy in range(top + 15, 370, 30):
            for wx in range(x + 10, x + 70, 22):
                cv2.rectangle(bg, (wx, wy), (wx + 12, wy + 14), (60, 70, 80), -1)
    for x in range(0, w, 40):
        cv2.line(bg, (x, 420), (x + 20, 540), (200, 200, 200), 3)
    return bg


def add_person(scene, cx, t=0.0):
    """Desenha uma pessoa (com textura de roupa) e devolve (frame, máscara 0/255)."""
    f = scene.copy()
    m = np.zeros(scene.shape[:2], np.uint8)
    for img, color in ((f, None), (m, 255)):
        cv2.circle(img, (cx, 170), 48, color or (80, 120, 190), -1)                      # cabeça
        cv2.ellipse(img, (cx, 340), (85, 140), 0, 0, 360, color or (40, 40, 160), -1)    # corpo
        cv2.ellipse(img, (cx - 110, 300), (25, 90), 20, 0, 360, color or (80, 120, 190), -1)  # braço
    for y in range(230, 470, 24):                                         # listras da roupa
        cv2.line(f, (cx - 70, y), (cx + 70, y + 10), (90, 90, 200), 4)
    noise = np.random.default_rng(int(t * 10)).normal(0, 2, f.shape)
    return np.clip(f + noise, 0, 255).astype(np.uint8), m


def test_segmenter(scene):
    seg = PersonSegmenter()
    prob = seg.probability(scene)
    seg.close()
    assert prob.shape == scene.shape[:2] and prob.dtype == np.float32
    assert prob.mean() < 0.2, prob.mean()     # cenário sem ninguém: quase nada é pessoa
    print(f"segmentador: ok (prob. média sem pessoas {prob.mean():.3f})")


def test_background(scene):
    """Aprende o fundo com uma pessoa andando na frente o tempo todo."""
    bgm = BackgroundModel()
    for i, cx in enumerate(range(250, 760, 15)):
        frame, m = add_person(scene, cx, i)
        bgm.update(frame, m)
    assert bgm.known_fraction > 0.97, bgm.known_fraction
    # onde a pessoa estava no começo, o fundo aprendido é o cenário, não a pessoa
    region = (slice(220, 450), slice(180, 320))
    err = np.abs(bgm.image()[region].astype(int) - scene[region].astype(int)).mean()
    assert err < 6, err

    # pessoa parada desde o primeiro frame: a área nunca vista é preenchida (inpaint)
    still = BackgroundModel()
    frame, m = add_person(scene, 480)
    still.update(frame, m)
    assert still.known_fraction < 0.9 and still.image() is not None
    print(f"fundo com pessoa na cena: ok ({bgm.known_fraction * 100:.0f}% aprendido, erro {err:.1f})")
    return bgm


def test_render(scene, bgm, out_dir=Path("tests/output")):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, mask = add_person(scene, 480)

    box = InvisibleBox(560, 300, 420, 300, angle=math.radians(-8))
    glass = GlassEffect()
    out = glass.render(frame, bgm.image(), mask, box.mask(frame.shape), box.bbox(), t=1.0)

    inside = box.mask(frame.shape) > 0
    body = (mask > 0) & inside
    # dentro da janela, o corpo deve ficar mais parecido com o fundo que com a pessoa
    err_bg = np.abs(out[body].astype(int) - scene[body].astype(int)).mean()
    err_person = np.abs(out[body].astype(int) - frame[body].astype(int)).mean()
    assert err_bg < err_person, (err_bg, err_person)
    # fora da janela nada muda
    assert np.array_equal(out[~inside], frame[~inside])

    from invisible_box import overlay
    c = BoxController(960, 540)
    c.box, c.state = box, State.FREE
    view = out.copy()
    overlay.draw_box(view, c, 1.0)
    overlay.draw_hud(view, c, 30.0, bgm.known_fraction)
    overlay.debug_panel(view, bgm.image(), mask)
    grid = np.vstack([np.hstack([frame, view])])
    cv2.imwrite(str(out_dir / "render.png"), grid)

    # medida de desempenho do efeito
    import time
    t0 = time.perf_counter()
    for i in range(30):
        glass.render(frame, bgm.image(), mask, box.mask(frame.shape, i / 30), box.bbox(), i / 30)
    ms = (time.perf_counter() - t0) / 30 * 1000
    print(f"render: ok (corpo vs fundo {err_bg:.1f} / vs pessoa {err_person:.1f}), {ms:.1f} ms por frame")


if __name__ == "__main__":
    test_gestures()
    test_controller()
    cena = synthetic_scene()
    test_segmenter(cena)
    fundo = test_background(cena)
    test_render(cena, fundo)
    print("TODOS OS TESTES PASSARAM")
