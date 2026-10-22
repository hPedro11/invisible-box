"""Baixa o modelo do MediaPipe HandLandmarker para a pasta models/ (mesmo passo do Lab13)."""
from pathlib import Path
from urllib.request import urlretrieve

from invisible_box import config


def main():
    destino = Path(config.MODEL_PATH)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        print(f"{destino} já existe.")
        return
    print(f"Baixando {destino.name}...")
    urlretrieve(config.MODEL_URL, destino)
    print(f"Pronto: {destino}")


if __name__ == "__main__":
    main()
