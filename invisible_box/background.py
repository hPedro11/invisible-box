"""Fundo da cena aprendido mesmo com pessoas na frente (Lab07 + Lab06).

Não existe mais a etapa "saia da frente da câmera". A cada frame, a segmentação
do MediaPipe diz onde estão as pessoas; tudo que NÃO é pessoa (com uma margem)
entra no fundo. Quando alguém se mexe, o pedaço de cenário que estava atrás
aparece e é aprendido. O que nunca foi visto (alguém parado o tempo todo) é
preenchido com inpaint a partir dos vizinhos até ser visto de verdade.

Exposição: a webcam clareia/escurece sozinha (escurece quando a pessoa sai e
só sobra parede branca). A cada atualização medimos, na parte visível do
fundo, quanto a câmera mudou (frame / fundo guardado, por canal) e corrigimos
o fundo INTEIRO por esse fator. Sem isso, o pedaço atrás da pessoa ficava
"congelado" na exposição antiga e aparecia como uma mancha cinza.

Proteção contra "fantasmas": se o segmentador falha (luz ruim, pele parecida
com a parede), um pedaço da pessoa podia entrar no fundo e ficar lá. Por isso
o fundo só aprende pixels (1) com chance baixíssima de ser pessoa, (2) PARADOS
(diferença entre frames, Lab07: gente sempre se mexe um pouco, parede não) e
(3) assim por BG_CONFIRM_FRAMES seguidos. E um pedaço visível e parado muito
diferente do guardado é trocado na hora: o fantasma some assim que aparece.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config


def _ellipse(size):
    size = max(int(size), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class BackgroundModel:
    def __init__(self):
        self._open_k = _ellipse(config.MORPH_OPEN_KERNEL)
        self._close_k = _ellipse(config.MORPH_CLOSE_KERNEL)
        # retângulo: a dilatação fica separável e bem mais rápida que com elipse
        self._margin_k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.PERSON_DILATE, config.PERSON_DILATE))
        self._executor = None   # thread do inpaint (criada só quando precisa)
        self.reset()

    def reset(self):
        self._bg = None       # float32, média do que já foi visto sem pessoa
        self._known = None    # uint8 0/255: pixels que já foram vistos sem pessoa
        self._streak = None   # uint8: frames seguidos em que o pixel está sem pessoa
        self._fill = None     # cache do inpaint das áreas nunca vistas
        self._pending = None  # inpaint rodando em paralelo
        self._image = None    # fundo final (uint8) usado pelo efeito
        self._frames = 0
        self._person_prev = None  # máscara anterior (para a histerese)
        self._gain = np.ones(3, np.float32)  # quanto a exposição mudou desde o início (B, G, R)
        self._prev_gray = None    # frame anterior reduzido (para detectar movimento)

    @property
    def ready(self):
        return self._image is not None

    @property
    def known_fraction(self):
        return 0.0 if self._known is None else cv2.countNonZero(self._known) / self._known.size

    def image(self):
        return self._image

    @property
    def exposure_gain(self):
        """Quanto a exposição da câmera mudou desde o início do fundo (B, G, R)."""
        return self._gain.copy()

    def unknown_mask(self):
        """255 onde o fundo nunca foi visto (preenchido por inpaint)."""
        return cv2.bitwise_not(self._known)

    def missing_fraction(self, region_mask):
        """Fração da região (ex.: a janela) cujo fundo ainda não foi visto."""
        total = cv2.countNonZero(region_mask)
        if self._known is None or total == 0:
            return 0.0
        return cv2.countNonZero(cv2.bitwise_and(region_mask, self.unknown_mask())) / total

    def person_mask(self, prob, shape):
        """Máscara binária (0/255) das pessoas no tamanho `shape` do frame.

        Limiar e morfologia rodam na resolução do segmentador (pequena, rápido);
        a ampliação é linear + novo limiar, o que deixa a borda lisa em vez de
        "em escadinha".

        Limiar duplo (histerese, como no Canny): o pixel vira pessoa acima de
        SEG_THRESHOLD, mas só deixa de ser abaixo de SEG_KEEP_THRESHOLD. A borda
        para de oscilar e regiões de pouca certeza (cabelo) continuam cobertas.
        """
        mask = (prob > config.SEG_THRESHOLD).astype(np.uint8) * 255   # Lab02: limiarização
        prev = self._person_prev
        if prev is not None and prev.shape == mask.shape:
            keep = (prob > config.SEG_KEEP_THRESHOLD).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(keep, prev))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)    # Lab06: tira pontinhos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)  # Lab06: fecha buracos
        self._person_prev = mask
        h, w = shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def update(self, frame, person_mask, prob=None):
        """Aprende o fundo só onde não há pessoa (com margem em volta delas).

        `prob` (probabilidade do segmentador, reduzida) deixa o aprendizado mais
        exigente: pixels com qualquer dúvida não entram no fundo.
        """
        if self._bg is None:
            self._bg = frame.astype(np.float32)
            self._known = np.zeros(frame.shape[:2], np.uint8)
            self._streak = np.zeros(frame.shape[:2], np.uint8)
        self._frames += 1

        # o fundo muda devagar: aprender a cada BG_UPDATE_EVERY frames basta
        if self._image is None or self._frames % config.BG_UPDATE_EVERY == 0:
            self._learn(frame, person_mask, prob)
        return self._image

    @staticmethod
    def _to_frame_size(mask, shape):
        h, w = shape[:2]
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _not_learnable(self, shape, small, person_mask, prob):
        """Onde o fundo NÃO pode aprender. Devolve (pessoa ou dúvida, + movimento)."""
        people = person_mask
        if prob is not None:
            doubt = cv2.compare(prob, float(config.SEG_LEARN_THRESHOLD), cv2.CMP_GT)
            people = cv2.bitwise_or(people, self._to_frame_size(doubt, shape))
        blocked = people

        # movimento: diferença entre frames em 1/4 da resolução (Lab07)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            moving = cv2.compare(cv2.absdiff(gray, self._prev_gray),
                                 float(config.BG_MOTION_THRESHOLD), cv2.CMP_GT)
            moving = cv2.dilate(moving, _ellipse(3))
            blocked = cv2.bitwise_or(blocked, self._to_frame_size(moving, shape))
        self._prev_gray = gray
        return people, blocked

    def _learn(self, frame, person_mask, prob=None):
        # uma cópia pequena do frame e do fundo (1/4) serve para movimento,
        # exposição e fantasmas; contas em resolução cheia só onde precisa
        h, w = frame.shape[:2]
        size_s = (max(w // 4, 1), max(h // 4, 1))
        small = cv2.resize(frame, size_s, interpolation=cv2.INTER_AREA)
        bg_small = cv2.resize(self._bg, size_s, interpolation=cv2.INTER_NEAREST)

        people, blocked = self._not_learnable(frame.shape, small, person_mask, prob)
        free = cv2.bitwise_not(cv2.dilate(blocked, self._margin_k))   # Lab05/Lab06

        # só confia no pixel depois de alguns frames seguidos sem pessoa: se o
        # segmentador falhar em um frame, um pedaço da pessoa não vira "fundo".
        cv2.add(self._streak, 1, dst=self._streak, mask=free)
        self._streak[free == 0] = 0
        confirm = max(1, -(-config.BG_CONFIRM_FRAMES // config.BG_UPDATE_EVERY))   # em atualizações
        stable = cv2.compare(self._streak, confirm, cv2.CMP_GE)

        # exposição: medida em todo fundo conhecido que não é pessoa, MESMO "se
        # mexendo" (uma mudança de exposição mexe a imagem inteira de uma vez)
        if config.BG_EXPOSURE_COMPENSATION and cv2.countNonZero(self._known):
            visible = cv2.bitwise_and(cv2.bitwise_not(cv2.dilate(people, self._margin_k)), self._known)
            visible_s = cv2.resize(visible, size_s, interpolation=cv2.INTER_NEAREST)
            ratio = self._exposure_ratio(small, bg_small, visible_s)
            if ratio is not None:
                # o fundo inteiro passa para a exposição atual (inclusive o pedaço escondido)
                scale = (float(ratio[0]), float(ratio[1]), float(ratio[2]), 0.0)
                cv2.multiply(self._bg, scale, dst=self._bg)
                bg_small = cv2.multiply(bg_small, scale)
                self._gain *= ratio

        # pixel confirmado pela primeira vez: copia direto (taxa 1.0); os outros:
        # média que acompanha a luz (Lab07)
        new = cv2.bitwise_and(stable, cv2.bitwise_not(self._known))
        # já conhecido, visível, parado e MUITO diferente do guardado: o guardado
        # estava errado (ex.: pedaço de pessoa que entrou por engano) -> troca na hora
        if config.BG_REPLACE_DIFF > 0 and cv2.countNonZero(self._known):
            changed = self._changed(small, bg_small, frame.shape)
            new = cv2.bitwise_or(new, cv2.bitwise_and(cv2.bitwise_and(stable, self._known), changed))
        cv2.accumulateWeighted(frame, self._bg, 1.0, mask=new)
        cv2.accumulateWeighted(frame, self._bg, config.BG_LEARNING_RATE, mask=stable)
        cv2.bitwise_or(self._known, stable, dst=self._known)

        image = cv2.convertScaleAbs(self._bg)
        unknown = cv2.bitwise_not(self._known)
        if cv2.countNonZero(unknown):
            self._refresh_fill(image, unknown)
            cv2.copyTo(self._fill, unknown, image)
        else:
            self._fill = None
        self._image = image

    def _refresh_fill(self, image, unknown):
        """Mantém o "chute" das áreas nunca vistas atualizado.

        O inpaint em resolução boa leva dezenas de ms; ele roda numa thread em
        paralelo (o OpenCV libera o Python enquanto calcula) e o resultado entra
        quando fica pronto. Até lá, vale o chute anterior. Só o primeiro é
        calculado na hora, porque ainda não há nenhum.
        """
        if self._pending is not None and self._pending.done():
            fill = self._pending.result()
            self._pending = None
            if fill.shape == image.shape:
                self._fill = fill
        if self._fill is None or self._fill.shape != image.shape:
            self._fill = self._inpaint(image, unknown)
        elif self._pending is None and self._frames % config.BG_INPAINT_EVERY == 0:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
            self._pending = self._executor.submit(self._inpaint, image.copy(), unknown.copy())

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _changed(self, small, bg_small, shape):
        """255 onde o frame difere muito do fundo guardado (calculado em 1/4)."""
        b, g, r = cv2.split(cv2.absdiff(small.astype(np.float32), bg_small))
        diff = cv2.max(cv2.max(b, g), r)          # maior diferença entre os canais
        changed = cv2.compare(diff, float(config.BG_REPLACE_DIFF), cv2.CMP_GT)
        changed = cv2.erode(changed, _ellipse(3))    # ignora pontinhos isolados de ruído
        return self._to_frame_size(changed, shape)

    def _exposure_ratio(self, small, bg_small, visible_s):
        """Quanto a câmera clareou/escureceu: frame / fundo na parte visível (por canal).

        Ignora pixels estourados ou quase pretos, onde a razão não diz nada.
        Razão das médias por canal (cv2.mean com máscara) = robusto a ruído. Devolve None se não há o que medir ou
        se a mudança é desprezível (aí nem mexe no fundo).
        """
        valid = cv2.inRange(small, (9, 9, 9), (246, 246, 246))   # nem preto, nem estourado
        m = cv2.bitwise_and(valid, visible_s)
        if cv2.countNonZero(m) < 200:          # pouco fundo visível: não dá para medir
            return None
        f_mean = np.array(cv2.mean(small, mask=m)[:3], np.float32)
        b_mean = np.array(cv2.mean(bg_small, mask=m)[:3], np.float32)
        ratio = f_mean / np.maximum(b_mean, 1.0)
        ratio = np.clip(ratio, 0.5, 2.0)
        # suaviza: aplica só parte da mudança por vez (evita tremer com ruído)
        ratio = 1.0 + (ratio - 1.0) * (1.0 - config.BG_GAIN_SMOOTHING)
        if np.abs(ratio - 1.0).max() < 0.002:
            return None
        return ratio.astype(np.float32)

    @staticmethod
    def _inpaint(image, unknown):
        """Preenche as áreas nunca vistas com os vizinhos (em escala menor, para ser rápido)."""
        h, w = unknown.shape
        sw = min(config.BG_INPAINT_WIDTH, w)
        sh = max(round(h * sw / w), 1)
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(unknown, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_mask = cv2.dilate(small_mask, _ellipse(3))
        filled = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
        return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
