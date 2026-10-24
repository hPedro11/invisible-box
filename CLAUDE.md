# CLAUDE.md

Contexto do projeto para o Claude Code. Responda sempre em português (pt-BR); código, comentários e textos da interface também em pt-BR.

## O que é

**Invisible Box**: projeto em grupo da matéria *Cognitive Computing, Computer Vision and IoT Systems* (FIAP), módulo de Processamento Digital de Imagens. Requisito do professor: usar o conteúdo dos Labs 01 a 14 do material da disciplina (https://arnaldojr.github.io/cognitivecomputing/), com foco em MediaPipe.

Uma janela de invisibilidade em tempo real pela webcam: o usuário aponta os dois indicadores para os cantos de um retângulo e salva com uma pinça rápida; dentro dele, as pessoas ficam transparentes, só com contornos de vidro que distorcem o fundo (efeito Predador). Referência visual: o vídeo do TikTok @3.27wx (TouchDesigner + MediaPipe). A janela é móvel: arrastar, redimensionar pelo canto, girar/escalar com duas mãos e arremessar com inércia.

Entregáveis pedidos: repositório no GitHub, aprovação da ideia pelo professor, README com descrição, diagrama (Mermaid) e integrantes. O README já tem tudo isso.

## Arquitetura (pacote `invisible_box/`)

- `main.py` e `calibrar.py`: só abrem a câmera/janela e tratam teclas; o processamento de cada frame fica em `pipeline.py` (`Pipeline.step`): flip -> mãos e segmentação -> máscara das pessoas -> atualiza o fundo -> controller -> efeito -> overlay. Não há etapa de calibração.
- Resolução: `open_camera` pede `FRAME_WIDTH`x`FRAME_HEIGHT` (1280x720) e usa o que a câmera entregar, SEM esticar (a webcam do Pedro não tem 960x540 e entregava 640x480, que era esticado). O MediaPipe recebe uma cópia reduzida a `PROCESS_WIDTH` (`imaging.reduced_copy`); landmarks são normalizados e a máscara volta ao tamanho cheio em `person_mask(prob, shape)`.
- `config.py`: todos os parâmetros ajustáveis (pinça, `SEG_THRESHOLD`, fundo, `GLASS_*`, atrito). Ajustes finos vão aqui. Os px de `GLASS_*` valem para `GLASS_REF_WIDTH` (960) e escalam com a resolução.
- `hand_tracker.py`: MediaPipe HandLandmarker, API Tasks em modo VIDEO (`detect_for_video`, timestamps crescentes, entrada RGB). Não usar a API antiga `mp.solutions`.
- `person_segmenter.py`: MediaPipe ImageSegmenter em modo VIDEO; devolve a probabilidade de pessoa (suavizada no tempo). A máscara (`BackgroundModel.person_mask`) usa limiar duplo/histerese: entra acima de `SEG_THRESHOLD`, só sai abaixo de `SEG_KEEP_THRESHOLD` (borda estável, cabelo coberto). Modelo padrão `selfie_segmenter` (~6 ms); `selfie_multiclass_256x256` tem bordas melhores mas leva ~90 ms. Com 1 máscara ela já é "pessoa"; com várias, pessoa = 1 - máscara 0.
- `gestures.py`: pinça = distância polegar-indicador / tamanho da mão, com histerese (`PINCH_ON`/`PINCH_OFF`) e eventos de borda. A pinça é aceita se um dos 3 últimos dedos está esticado OU (no controller) se o indicador esteve esticado nos últimos `POINTING_PINCH_WINDOW` s, para a pinça a partir do "apontar" funcionar sem confundir com punho.
- `background.py`: fundo aprendido COM pessoas na cena: copia na primeira vez que um pixel fica `BG_CONFIRM_FRAMES` frames seguidos livre (não é pessoa, prob. < `SEG_LEARN_THRESHOLD`, e PARADO: diferença entre frames < `BG_MOTION_THRESHOLD`; tudo dilatado por `PERSON_DILATE`), depois `accumulateWeighted`. Isso evita "fantasmas": num ambiente com luz ruim o segmentador perdia o rosto e ele era gravado no fundo. Pixel visível e parado muito diferente do guardado (`BG_REPLACE_DIFF`) é trocado na hora (o fantasma some quando a área aparece). O ganho de exposição é medido em todo fundo conhecido sem pessoa, mesmo "em movimento" (mudança de exposição mexe a imagem toda); áreas nunca vistas são preenchidas com `cv2.inpaint` em `BG_INPAINT_WIDTH` (480 px) numa thread em paralelo (`ThreadPoolExecutor`; só o primeiro é síncrono), a cada `BG_INPAINT_EVERY` frames. Esse "chute" é o que fica borrado atrás de quem nunca saiu do lugar. `known_fraction` aparece no HUD; `missing_fraction(janela)` dispara o aviso. Compensa a exposição automática da webcam: o fundo fica guardado numa exposição de referência e é mostrado x ganho atual por canal (medido na parte visível do fundo); sem isso, o pedaço atrás da pessoa ficava cinza (aprendido quando a câmera estava mais escura). Aprende a cada `BG_UPDATE_EVERY` frames; usa `cv2.accumulateWeighted`/`cv2.copyTo` com máscara (o `np.copyto(where=...)` custava ~5 ms em 720p).
- `glass.py`: relevo = borda arredondada a partir do `distanceTransform` da silhueta INTEIRA (`GLASS_BEVEL`; a janela só recorta no fim) + linhas internas longas (Canny em meia resolução, contornos com `arcLength` >= `GLASS_RIDGE_MIN_LENGTH`, com média móvel `GLASS_RIDGE_MEMORY` em coordenadas do frame para não piscar) -> Sobel -> um `cv2.remap` por canal (`GLASS_CHROMA`) + reflexo de luz direcional só nos pontos virados para a luz (sem contorno branco); a transparência cobre `GLASS_COVER` px além da borda. Teste de estabilidade (ruído de câmera e de segmentação): tremido entre frames caiu de 3,0 para 1,1. Miolo plano = fundo limpo. Processa só o retângulo da pessoa dentro da janela; todos os mapas suaves (inclinação, ondulação, brilho, sombra, alpha já recortado pela janela) são calculados em `GLASS_RELIEF_SCALE` e ampliados; em resolução cheia ficam só os `remap` e os `cv2.blendLinear` (float de 3 canais no NumPy era o gargalo).
- `box.py`: geometria (centro, w, h, ângulo), cantos, redimensionar pelo canto oposto, física de inércia/quique, ondulação.
- `controller.py`: máquina de estados DEFININDO -> LIVRE <-> ARRASTANDO / REDIMENSIONANDO / GIRANDO/ESCALANDO / DESLIZANDO. Em DEFININDO, a prévia vai de ponta de indicador a ponta de indicador; a pinça rápida (fecha e abre em até `TAP_MAX_SECONDS`) salva os cantos de `TAP_LOOKBACK` s antes da pinça, porque o dedo se move ao fazer a pinça.
- `overlay.py`: janela = linha fina e translúcida como no vídeo (`BOX_*` no config; cantos só aparecem com a mão perto), prévia no mesmo estilo, HUD (com % do fundo aprendido) e painel de debug (tecla D).

## Comandos

- Instalar: `pip install -r requirements.txt` e `python download_models.py` (baixa `models/hand_landmarker.task`, que não é versionado).
- Rodar: `python main.py` (`--camera 1`, `--debug`).
- Calibrar com a webcam: `python calibrar.py` (sliders de segmentação/pinça/vidro + gravações rotuladas da pinça; `S` salva medições e sugestões em `calibracao.json`, não versionado).
- Testes sem webcam: `python -m tests.test_offline` (mãos e cena sintéticas; salva `tests/output/render.png`).

## Estado atual

- Segunda versão (branch `calibracao-webcam`): interação por indicadores + pinça rápida, segmentação do MediaPipe no lugar da subtração de fundo, fundo aprendido com pessoas na cena e vidro transparente com aberração cromática. Testada com dados sintéticos e com um frame real do vídeo de referência. **Ainda não foi testada com a webcam real**; próximo passo: rodar `calibrar.py` e ajustar `PINCH_*`, `SEG_THRESHOLD` e `GLASS_*`.
- A câmera precisa ficar parada (o fundo aprendido vale para aquele ponto de vista).
- Limite físico: quem fica parado no mesmo lugar desde o início deixa um buraco nunca visto no fundo (inpaint = mancha lisa). O primeiro teste com webcam mostrou isso (fundo 93%). Um aviso no topo da janela (`overlay.draw_missing_notice`) mostra a % do fundo dela nunca vista quando passa de `BG_MISSING_WARN`, e o debug (D) pinta essas áreas de magenta; a solução é a pessoa sair do quadro por ~2 s.
- Repositório: https://github.com/hPedro11/invisible-box (`main` + tags `v1.0` e `v2.0`; cada versão marcante ganha uma tag nova). Falta: enviar a ideia para aprovação do professor.

## Grupo

Pedro Henrique Lima (RM553664), Murilo Fernandes (RM553582), Thais Mari Costa (RM553620), Bruno Furlan (RM553604), Maria Alice Sousa (RM552717).
