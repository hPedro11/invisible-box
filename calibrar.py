"""Calibração do Invisible Box com a webcam real.

Roda o mesmo pipeline do main.py, com sliders para os parâmetros principais e
medições ao vivo. Também grava amostras rotuladas da pinça e sugere valores
para o config.py a partir delas.

Uso:
    python calibrar.py             # webcam padrão
    python calibrar.py --camera 1

Teclas:
    1  grava mãos SEM pinça (apontando, abertas, relaxadas)
    2  grava pinças rápidas (indicador no polegar e solta)  -> 1 + 2 sugerem PINCH_ON / PINCH_OFF
    S  salva sliders, medições e sugestões em calibracao.json
    R  reaprende o fundo | N nova janela | D debug | H mãos | Q/Esc sair
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from invisible_box import config, gestures
from invisible_box.pipeline import Pipeline, open_camera
from main import check_models, create_window

WINDOW = "Invisible Box - calibracao"
SLIDERS = "Ajustes"
OUTPUT = Path("calibracao.json")
FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW = (0, 230, 255)

# nome do slider: (atributo do config, máximo do slider, escala slider -> valor)
TRACKBARS = {
    "SEG_THRESHOLD x100": ("SEG_THRESHOLD", 100, 100),
    "SEG_KEEP_THRESHOLD x100": ("SEG_KEEP_THRESHOLD", 100, 100),
    "PINCH_ON x100": ("PINCH_ON", 100, 100),
    "PINCH_OFF x100": ("PINCH_OFF", 100, 100),
    "GLASS_BEVEL": ("GLASS_BEVEL", 40, 1),
    "GLASS_RIDGES x10": ("GLASS_RIDGES", 30, 10),
    "GLASS_RIDGE_MIN_LENGTH": ("GLASS_RIDGE_MIN_LENGTH", 500, 1),
    "GLASS_RIDGE_MEMORY x100": ("GLASS_RIDGE_MEMORY", 95, 100),
    "GLASS_COVER": ("GLASS_COVER", 30, 1),
    "GLASS_STRENGTH": ("GLASS_STRENGTH", 60, 1),
    "GLASS_CHROMA x100": ("GLASS_CHROMA", 100, 100),
    "GLASS_SHIMMER x10": ("GLASS_SHIMMER", 100, 10),
    "GLASS_HIGHLIGHT x100": ("GLASS_HIGHLIGHT", 100, 100),
}

RECORDINGS = {  # tecla: (nome, instrução, contagem regressiva em s, frames gravados)
    ord("1"): ("aberta", "Maos SEM pinca: apontando, abertas, relaxadas", 2.0, 120),
    ord("2"): ("pinca", "Pincas rapidas: indicador no polegar e solta, varias vezes", 2.0, 120),
}


def parse_args():
    p = argparse.ArgumentParser(description="Calibração do Invisible Box")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    p.add_argument("--no-mirror", action="store_true")
    return p.parse_args()


def create_sliders():
    cv2.namedWindow(SLIDERS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SLIDERS, 480, 560)
    for name, (attr, maximum, scale) in TRACKBARS.items():
        value = int(round(getattr(config, attr) * scale))
        cv2.createTrackbar(name, SLIDERS, min(value, maximum), maximum, lambda _v: None)


def read_sliders(controller):
    """Copia os sliders para o config (os módulos leem o config a cada frame)."""
    for name, (attr, _maximum, scale) in TRACKBARS.items():
        v = cv2.getTrackbarPos(name, SLIDERS)
        setattr(config, attr, v / scale if scale != 1 else v)
    config.GLASS_STRENGTH = float(config.GLASS_STRENGTH)
    if config.PINCH_OFF <= config.PINCH_ON:  # a histerese precisa de OFF > ON
        config.PINCH_OFF = config.PINCH_ON + 0.01
    controller._pinch.on, controller._pinch.off = config.PINCH_ON, config.PINCH_OFF


def current_values():
    return {attr: getattr(config, attr) for attr, _m, _s in TRACKBARS.values()}


class Recorder:
    """Grava amostras rotuladas da razão da pinça."""

    def __init__(self):
        self.ratios = {"aberta": [], "pinca": []}
        self.active = None       # [nome, instrução, início da gravação, frames restantes]

    def start(self, key, now):
        name, text, countdown, frames = RECORDINGS[key]
        self.active = [name, text, now + countdown, frames]

    def status(self, now):
        if self.active is None:
            return None
        name, text, t_start, left = self.active
        if now < t_start:
            return f"{text} - comeca em {t_start - now:.1f}s"
        return f"Gravando '{name}': {left} frames  ({text})"

    def feed(self, now, hands):
        if self.active is None or now < self.active[2]:
            return
        name = self.active[0]
        for h in hands:
            self.ratios[name].append(gestures.pinch_ratio(h.points))
        self.active[3] -= 1
        if self.active[3] <= 0:
            print(f"[calibracao] gravação '{name}' concluída")
            self.active = None

    def summary(self):
        out = {}
        for name, vals in self.ratios.items():
            if vals:
                a = np.asarray(vals)
                out[name] = {
                    "amostras": len(a),
                    "p5": round(float(np.percentile(a, 5)), 3),
                    "p10": round(float(np.percentile(a, 10)), 3),
                    "mediana": round(float(np.median(a)), 3),
                    "p95": round(float(np.percentile(a, 95)), 3),
                }
        return out

    @staticmethod
    def suggestions(summary):
        sug, notes = {}, []
        if "aberta" in summary and "pinca" in summary:
            # nas pinças rápidas metade dos frames é o dedo indo/voltando, então o
            # p10 representa os frames com a pinça de fato fechada
            closed, opened = summary["pinca"]["p10"], summary["aberta"]["p5"]
            gap = opened - closed
            if gap <= 0.05:
                notes.append(f"Pinca e mao aberta se sobrepoem (pinca p10={closed}, "
                             f"aberta p5={opened}). Regrave ou aproxime as maos da camera.")
            else:
                sug["PINCH_ON"] = round(closed + 0.35 * gap, 2)
                sug["PINCH_OFF"] = round(closed + 0.65 * gap, 2)
        return sug, notes


def draw_measurements(view, controller, person, known, status):
    """Painel à esquerda: razão da pinça por mão, área das pessoas e fundo aprendido."""
    x0, y0, bar_w = 10, 44, 260
    lines = 2 + 2 * max(len(controller.hands), 1)
    cv2.rectangle(view, (0, y0 - 4), (x0 + bar_w + 140, y0 + 22 * lines + 8), (0, 0, 0), -1)
    pct = 100 * float((person > 0).mean())
    cv2.putText(view, f"pessoas: {pct:5.1f}% da imagem   fundo: {known * 100:3.0f}%",
                (x0, y0 + 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y = y0 + 34
    if not controller.hands:
        cv2.putText(view, "nenhuma mao detectada", (x0, y + 10), FONT, 0.45, (160, 160, 160), 1)
    for h in controller.hands:
        r = gestures.pinch_ratio(h.points)
        color = (120, 255, 120) if h.pinch.pinching else (255, 255, 255)
        cv2.putText(view, f"{h.key}: razao {r:.2f}  {'PINCA' if h.pinch.pinching else ''}",
                    (x0, y + 10), FONT, 0.45, color, 1, cv2.LINE_AA)
        by = y + 16
        cv2.rectangle(view, (x0, by), (x0 + bar_w, by + 10), (80, 80, 80), 1)
        cv2.rectangle(view, (x0, by), (x0 + int(min(r, 1.0) * bar_w), by + 10), color, -1)
        for thr, c in ((config.PINCH_ON, (120, 255, 120)), (config.PINCH_OFF, (40, 170, 255))):
            tx = x0 + int(thr * bar_w)
            cv2.line(view, (tx, by - 3), (tx, by + 13), c, 2)
        y += 44
    if status:
        cv2.putText(view, status, (x0, view.shape[0] - 40), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(view, "[1] sem pinca [2] pincas rapidas [S] salvar", (x0, y + 4),
                FONT, 0.45, YELLOW, 1, cv2.LINE_AA)


def save(recorder):
    summary = recorder.summary()
    sug, notes = recorder.suggestions(summary)
    data = {"valores_atuais": current_values(), "medicoes": summary,
            "sugestoes": sug, "observacoes": notes}
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[calibracao] salvo em {OUTPUT.resolve()}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    check_models()
    cap, size = open_camera(args.camera, args.width, args.height)
    pipe = Pipeline(size, mirror=config.MIRROR and not args.no_mirror, show_debug=True)
    recorder = Recorder()
    create_window(WINDOW, size)
    create_sliders()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler a câmera.")
                break
            read_sliders(pipe.controller)
            view = pipe.step(frame)
            recorder.feed(pipe.now, pipe.controller.hands)
            draw_measurements(view, pipe.controller, pipe.person, pipe.background.known_fraction,
                              recorder.status(pipe.now))
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in RECORDINGS:
                recorder.start(key, pipe.now)
            elif key == ord("s"):
                save(recorder)
            elif not pipe.handle_key(key):
                break
    finally:
        save(recorder)
        pipe.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
