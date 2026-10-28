# Invisible Box – Janela de Invisibilidade com Visão Computacional

Projeto da disciplina **Cognitive Computing, Computer Vision and IoT Systems** (FIAP), módulo de **Processamento Digital de Imagens**.

## Descrição do projeto

O **Invisible Box** é uma aplicação em tempo real que usa a webcam para criar uma "janela de invisibilidade" controlada pelas mãos.

O usuário marca um retângulo no ar apontando com os dois indicadores e salva a área com uma pinça rápida. Dentro dessa área, qualquer pessoa que passar fica transparente: em vez da pessoa, aparece o fundo da cena, e o corpo vira apenas contornos de vidro que distorcem o que está atrás, como no efeito do filme *Predador*.

A janela não é estática: ela pode ser **arrastada, redimensionada, girada e arremessada** com gestos de pinça, em tempo real.

### Funcionalidades

- **Criar a janela com as mãos:** a ponta de cada indicador marca um canto do retângulo; uma pinça rápida (indicador encosta no polegar e solta) salva a área.
- **Detecção de pessoas:** segmentação de pessoas do MediaPipe (`ImageSegmenter`), limpa com limiarização e morfologia. Funciona com várias pessoas e com pessoas paradas.
- **Fundo aprendido com gente na cena:** a cada frame, tudo que com certeza não é pessoa e está parado (diferença entre frames) entra no fundo; assim, se o segmentador falhar por um momento, a pessoa não é gravada no cenário. Se um pedaço do fundo guardado estiver errado, ele é corrigido assim que aparecer; quando alguém se mexe, o cenário que estava atrás é aprendido. O que nunca foi visto é preenchido com *inpainting* até aparecer. O fundo também acompanha o ajuste automático de exposição da webcam, para a parte escondida atrás da pessoa não ficar mais clara ou mais escura que o resto.
- **Efeito de vidro transparente:** a pessoa é trocada pelo fundo e só sobra a refração. O relevo do vidro tem uma borda arredondada em volta do corpo (`distanceTransform`) e filetes internos nos contornos longos da pessoa, como dedos, braços e rosto (Canny + contornos filtrados por comprimento). O Sobel desse relevo vira mapa de deslocamento (`cv2.remap`), cada canal de cor é deslocado com força diferente (aberração cromática) e um reflexo de luz dá volume ao vidro.
- **Manipulação da janela por gestos:**
  - pinça dentro da janela → arrastar;
  - pinça em um canto → redimensionar;
  - pinça com as duas mãos → girar e escalar;
  - soltar em movimento → a janela desliza com inércia e quica nas bordas.

### Técnicas e laboratórios utilizados

| Técnica | Laboratório |
|---|---|
| Manipulação de pixels, canais e desenho (`line`, `putText`, `fillPoly`) | Lab01 |
| Limiarização da máscara de pessoas | Lab02 |
| Filtros de convolução: Gaussian Blur, Sobel e Canny | Lab04 |
| Máscaras, operações bitwise, canais de cor e contornos (`findContours`) | Lab05 |
| Morfologia (abertura, fechamento e dilatação) | Lab06 |
| Modelo de fundo acumulado (`accumulateWeighted`) | Lab07 |
| Seleção de região (ROI) por interação do usuário: no Lab11 com eventos do mouse, aqui com os dedos | Lab11 |
| Razão de distâncias entre landmarks com limiar, confirmada no tempo (como o detector de fadiga do Lab12, aplicada à pinça) | Lab12 |
| Landmarks das mãos (HandLandmarker) e segmentação de pessoas (ImageSegmenter) | Lab13 |

### Tecnologias

- Python 3
- OpenCV
- MediaPipe (Tasks API – HandLandmarker e ImageSegmenter)
- NumPy

## Diagrama de funcionamento

### Pipeline por frame

```mermaid
flowchart TD
    A[Webcam] --> B[Captura do frame]
    B --> C[Flip horizontal]
    C --> D[MediaPipe HandLandmarker]
    C --> E[MediaPipe ImageSegmenter]

    D --> F[Reconhecimento de gestos<br/>indicador, pinça rápida, mão aberta]
    F --> G[Máquina de estados da janela]
    G --> H[Posição, tamanho e rotação da janela]
    H --> I[Máscara da janela - fillPoly]

    E --> K[Limiarização + morfologia]
    K --> L[Máscara das pessoas]
    C --> J
    L --> J[Fundo aprendido fora das pessoas e parado<br/>accumulateWeighted + inpaint]

    I --> M[Interseção das máscaras - bitwise_and]
    L --> M
    M --> N[Relevo: borda distanceTransform + contornos longos Canny<br/>Sobel]
    J --> O
    N --> O[Distorção do fundo por canal - remap]
    O --> P[Composição final + HUD]
    P --> Q[Exibição na tela]
```

### Estados da janela

```mermaid
stateDiagram-v2
    [*] --> Definindo
    Definindo --> Livre: indicadores nos cantos + pinça rápida
    Livre --> Arrastando: pinça dentro da janela
    Livre --> Redimensionando: pinça em um canto
    Livre --> Girando: pinça com as duas mãos
    Arrastando --> Girando: segunda mão faz pinça
    Arrastando --> Inercia: soltou a pinça em movimento
    Arrastando --> Livre: soltou a pinça devagar
    Redimensionando --> Livre: soltou a pinça
    Girando --> Arrastando: uma das mãos soltou
    Girando --> Livre: soltou as duas pinças
    Inercia --> Livre: velocidade zerou
    Inercia --> Arrastando: pinça na janela
    Livre --> Definindo: duas mãos abertas por ~1 s
```

A tecla `N` volta para *Definindo* a partir de qualquer estado; o gesto das duas mãos abertas só vale com a janela parada (*Livre*).

## Como executar

Requisitos: Python 3.10 ou mais novo (testado no 3.10 e no 3.14), Git e uma webcam. **A câmera precisa ficar parada** (apoiada ou em tripé), porque o efeito usa o fundo aprendido daquele ponto de vista.

```bash
git clone https://github.com/hPedro11/invisible-box.git
cd invisible-box
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py      # baixa os modelos de mãos e de segmentação (uma vez)
python main.py                 # use --camera 1 para outra câmera, --debug para ver fundo e máscara
```

O programa pede 1280x720 à câmera e usa a resolução que ela entregar, sem esticar a imagem (a resolução usada aparece no terminal). Para manter o FPS, o MediaPipe recebe uma cópia reduzida do frame e os mapas suaves do efeito de vidro são calculados em meia resolução.

### Como usar

1. **Fundo:** o HUD mostra quanto do fundo já foi visto (`fundo 87%`). Pessoas que passam ou se mexem não atrapalham, mas a câmera não enxerga atrás de quem está parado: saia do quadro por ~2 segundos uma vez para o cenário atrás de você ser aprendido. Enquanto faltar fundo dentro da janela (essa parte fica borrada), aparece no topo dela um aviso com a porcentagem que falta, e o painel de debug (`D`) pinta essas áreas de magenta.
2. **Criar a janela:** aponte os dois indicadores para os cantos desejados (aparece um retângulo branco) e faça uma pinça rápida com qualquer uma das mãos: o indicador encosta no polegar e solta. A janela é salva com os cantos de logo antes da pinça.
3. **Mover:** pinça dentro da janela e arraste. Solte com a mão em movimento para arremessar.
4. **Redimensionar:** pinça em um dos cantos.
5. **Girar e escalar:** pinça com as duas mãos dentro da janela e gire/afaste as mãos.
6. **Recriar:** com a janela parada, mostre as duas mãos abertas por ~1 segundo (ou tecle `N`).

| Tecla | Ação |
|---|---|
| `N` | nova janela |
| `R` | reaprender o fundo do zero |
| `D` | mostrar fundo aprendido e máscara das pessoas |
| `H` | mostrar/ocultar o esqueleto das mãos |
| `S` | salvar captura em `capturas/` |
| `Q` / `Esc` | sair |

Os parâmetros (limiares da pinça, sensibilidade da máscara, força do vidro, atrito etc.) ficam em `invisible_box/config.py`.

Para calibrar com a sua webcam, rode `python calibrar.py`: ele abre o app com sliders dos parâmetros principais (segmentação, pinça e vidro), mostra a razão da pinça de cada mão e grava amostras (teclas `1` mãos sem pinça, `2` pinças rápidas). A tecla `S` salva medições e valores sugeridos em `calibracao.json`.

### Testes sem webcam

```bash
python -m tests.test_offline
```

Simula mãos (landmarks sintéticos) e uma cena com uma pessoa andando para testar gestos, criação da janela com pinça rápida, máquina de estados, fundo aprendido com a pessoa na cena e o efeito de vidro. O resultado visual é salvo em `tests/output/render.png`:

![Teste sintético do efeito](docs/teste_sintetico.png)

## Estrutura do projeto

```
invisible-box/
├── main.py                  # abre a câmera e a janela; teclas
├── download_models.py       # baixa os modelos do MediaPipe
├── requirements.txt
├── invisible_box/
│   ├── config.py            # todos os parâmetros ajustáveis
│   ├── pipeline.py          # um frame: visão -> fundo -> gestos -> efeito -> interface
│   ├── imaging.py           # cópia reduzida do frame para o MediaPipe
│   ├── hand_tracker.py      # MediaPipe HandLandmarker em modo VIDEO (Lab13)
│   ├── person_segmenter.py  # MediaPipe ImageSegmenter: máscara das pessoas (Lab13)
│   ├── gestures.py          # pinça com histerese, dedo esticado, mão aberta
│   ├── background.py        # fundo aprendido com pessoas na cena + inpaint (Lab07, Lab06)
│   ├── glass.py             # vidro transparente: relevo + Sobel + remap por canal (Lab04, Lab05)
│   ├── box.py               # geometria da janela: girar, redimensionar, inércia
│   ├── controller.py        # máquina de estados dos gestos
│   └── overlay.py           # linha da janela, HUD e painel de debug (Lab01)
├── calibrar.py              # sliders + medições para ajustar o config com a webcam
├── models/                  # modelos do MediaPipe (não versionados)
└── tests/test_offline.py    # testes sem webcam
```

## Integrantes

| Nome | RM |
|---|---|
| Pedro Henrique Lima | RM553664 |
| Murilo Fernandes | RM553582 |
| Thais Mari Costa | RM553620 |
| Bruno Furlan | RM553604 |
| Maria Alice Sousa | RM552717 |
