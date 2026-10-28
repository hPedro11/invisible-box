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
        # baixa num arquivo temporário e só renomeia no fim: se o download cair,
        # não fica um modelo pela metade que pareceria "já baixado"
        parcial = destino.with_name(destino.name + ".part")
        try:
            urlretrieve(url, parcial)
            parcial.replace(destino)
        finally:
            parcial.unlink(missing_ok=True)
        print(f"Pronto: {destino}")


if __name__ == "__main__":
    main()
