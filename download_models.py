"""Baixa os modelos do MediaPipe para a pasta models/ (mesmo passo do Lab13).

- HandLandmarker: landmarks das mãos (gestos).
- ImageSegmenter (selfie_segmenter): máscara das pessoas (efeito e fundo).
"""
from pathlib import Path
from urllib.request import urlretrieve

from invisible_box import config

MODELOS = [
    (config.MODEL_PATH, config.MODEL_URL),
    (config.SEGMENTER_PATH, config.SEGMENTER_URL),
]


def main():
    for caminho, url in MODELOS:
        destino = Path(caminho)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            print(f"{destino} já existe.")
            continue
        print(f"Baixando {destino.name}...")
        urlretrieve(url, destino)
        print(f"Pronto: {destino}")


if __name__ == "__main__":
    main()
