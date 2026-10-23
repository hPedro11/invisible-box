# CLAUDE.md

Contexto do projeto para o Claude Code. Responda sempre em português (pt-BR); código, comentários e textos da interface também em pt-BR.

## O que é

**Invisible Box**: projeto em grupo da matéria *Cognitive Computing, Computer Vision and IoT Systems* (FIAP), módulo de Processamento Digital de Imagens. Requisito do professor: usar o conteúdo dos Labs 01 a 14 do material da disciplina (https://arnaldojr.github.io/cognitivecomputing/), com foco em MediaPipe.

Uma janela de invisibilidade em tempo real pela webcam: o usuário aponta os dois indicadores para os cantos de um retângulo e salva com uma pinça rápida; dentro dele, as pessoas ficam transparentes, só com contornos de vidro que distorcem o fundo (efeito Predador). Referência visual: o vídeo do TikTok @3.27wx (TouchDesigner + MediaPipe). A janela é móvel: arrastar, redimensionar pelo canto, girar/escalar com duas mãos e arremessar com inércia.

Entregáveis pedidos: repositório no GitHub, aprovação da ideia pelo professor, README com descrição, diagrama (Mermaid) e integrantes. O README já tem tudo isso.

## Arquitetura (pacote `invisible_box/`)

- `main.py`: loop principal. Câmera -> flip -> mãos e segmentação (no frame ORIGINAL) -> máscara das pessoas -> atualiza o fundo -> controller -> efeito -> overlay. Não há etapa de calibração.
- `config.py`: todos os parâmetros ajustáveis (pinça, `SEG_THRESHOLD`, fundo, `GLASS_*`, atrito). Ajustes finos vão aqui.
- `hand_tracker.py`: MediaPipe HandLandmarker, API Tasks em modo VIDEO (`detect_for_video`, timestamps crescentes, entrada RGB). Não usar a API antiga `mp.solutions`.
- `person_segmenter.py`: MediaPipe ImageSegmenter em modo VIDEO; devolve a probabilidade de pessoa (suavizada no tempo). Modelo padrão `selfie_segmenter` (~6 ms); `selfie_multiclass_256x256` tem bordas melhores mas leva ~90 ms. Com 1 máscara ela já é "pessoa"; com várias, pessoa = 1 - máscara 0.
- `gestures.py`: pinça = distância polegar-indicador / tamanho da mão, com histerese (`PINCH_ON`/`PINCH_OFF`) e eventos de borda. A pinça é aceita se um dos 3 últimos dedos está esticado OU (no controller) se o indicador esteve esticado nos últimos `POINTING_PINCH_WINDOW` s, para a pinça a partir do "apontar" funcionar sem confundir com punho.
- `background.py`: fundo aprendido COM pessoas na cena: copia na primeira vez que um pixel fica `BG_CONFIRM_FRAMES` frames seguidos sem pessoa (máscara dilatada por `PERSON_DILATE`), depois `accumulateWeighted`; áreas nunca vistas são preenchidas com `cv2.inpaint` em escala reduzida a cada `BG_INPAINT_EVERY` frames. `known_fraction` aparece no HUD; `missing_fraction(janela)` dispara o aviso.
- `glass.py`: relevo = borda arredondada a partir do `distanceTransform` da silhueta INTEIRA (`GLASS_BEVEL`; a janela só recorta no fim) + linhas internas longas (Canny em meia resolução, contornos com `arcLength` >= `GLASS_RIDGE_MIN_LENGTH`) -> Sobel -> um `cv2.remap` por canal (`GLASS_CHROMA`) + reflexo de luz direcional. Miolo plano = fundo limpo. Processa só o retângulo da pessoa dentro da janela e mistura com `cv2.blendLinear` (float de 3 canais no NumPy era o gargalo).
- `box.py`: geometria (centro, w, h, ângulo), cantos, redimensionar pelo canto oposto, física de inércia/quique, ondulação.
- `controller.py`: máquina de estados DEFININDO -> LIVRE <-> ARRASTANDO / REDIMENSIONANDO / GIRANDO/ESCALANDO / DESLIZANDO. Em DEFININDO, a prévia vai de ponta de indicador a ponta de indicador; a pinça rápida (fecha e abre em até `TAP_MAX_SECONDS`) salva os cantos de `TAP_LOOKBACK` s antes da pinça, porque o dedo se move ao fazer a pinça.
- `overlay.py`: contorno com brilho, prévia, HUD (com % do fundo aprendido) e painel de debug (tecla D).

## Comandos

- Instalar: `pip install -r requirements.txt` e `python download_models.py` (baixa `models/hand_landmarker.task`, que não é versionado).
- Rodar: `python main.py` (`--camera 1`, `--debug`).
- Calibrar com a webcam: `python calibrar.py` (sliders de segmentação/pinça/vidro + gravações rotuladas da pinça; `S` salva medições e sugestões em `calibracao.json`, não versionado).
- Testes sem webcam: `python -m tests.test_offline` (mãos e cena sintéticas; salva `tests/output/render.png`).

## Estado atual

- Segunda versão (branch `calibracao-webcam`): interação por indicadores + pinça rápida, segmentação do MediaPipe no lugar da subtração de fundo, fundo aprendido com pessoas na cena e vidro transparente com aberração cromática. Testada com dados sintéticos e com um frame real do vídeo de referência. **Ainda não foi testada com a webcam real**; próximo passo: rodar `calibrar.py` e ajustar `PINCH_*`, `SEG_THRESHOLD` e `GLASS_*`.
- A câmera precisa ficar parada (o fundo aprendido vale para aquele ponto de vista).
- Limite físico: quem fica parado no mesmo lugar desde o início deixa um buraco nunca visto no fundo (inpaint = mancha lisa). O primeiro teste com webcam mostrou isso (fundo 93%). O HUD avisa quando mais de `BG_MISSING_WARN` da janela não tem fundo conhecido, e o debug (D) pinta essas áreas de magenta; a solução é a pessoa sair do quadro por ~2 s.
- Repositório: https://github.com/hPedro11/invisible-box (`main` + tags `v1.0` e `v2.0`; cada versão marcante ganha uma tag nova). Falta: enviar a ideia para aprovação do professor.

## Grupo

Pedro Henrique Lima (RM553664), Murilo Fernandes (RM553582), Thais Mari Costa (RM553620), Bruno Furlan (RM553604), Maria Alice Sousa (RM552717).
