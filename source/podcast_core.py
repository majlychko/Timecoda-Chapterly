# алгоритмы глав, метрики и экономика. здесь нам не нужны веса нейросетей, поэтому модуль импортируется быстро
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# версия кода входит в ключи кэша: если мы меняем логику, старые результаты не подхватятся
VERSION = "2026-11-24.2"
# один seed на весь проект, чтобы случайные baseline и бутстреп повторялись
SEED = 42
# стоп-слова для tf-idf: служебные слова и разговорные паразиты, которые не говорят о теме
STOP = set("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну ли если уже или ни быть был него до вас вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет тогда кто этот того потому этого какой ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много три эту моя хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это вообще просто очень типа короче значит знаете сказать который которая которые давайте сегодня будем говорит потому поэтому значит собственно например самом деле понятно слушайте так далее спасибо привет всем добрый время суток".split())


def digest(value) -> str:
    # делаем короткий хэш от любых параметров, он служит именем файла в кэше
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:20]


def save_json(path, value):
    # сначала пишем во временный файл и только потом переименовываем,
    # чтобы при обрыве сессии в кэше не остался битый json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(p)


def timestamp(seconds):
    # переводим секунды в таймкод, который понимает youtube: мм:сс или чч:мм:сс
    n = max(0, int(seconds))
    h, rem = divmod(n, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def parse_timestamp(value):
    # обратная операция: таймкод из описания выпуска в секунды, с проверкой формата
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts):
        raise ValueError(f"Некорректный таймкод: {value}")
    nums = list(map(int, parts))
    # минуты и секунды не могут быть больше 59, иначе это опечатка в описании
    if any(x >= 60 for x in nums[1:]):
        raise ValueError(f"Некорректный таймкод: {value}")
    return float(sum(x * 60 ** i for i, x in enumerate(reversed(nums))))


def parse_chapters(description):
    # достаём авторские главы из html-описания выпуска
    # сначала превращаем переносы и пункты списка в строки, потом убираем все теги
    text = re.sub(r"<br\s*/?>|</li>|</p>", "\n", description, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]*>", "", text))
    out = {}
    for line in text.splitlines():
        # у радио-т встречаются два формата: «чч:мм:сс тема» и «тема - чч:мм:сс»
        m = re.match(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)\s*$", line)
        tail = re.match(r"^\s*(.+?)\s+[-\u2013\u2014]\s+(\d{1,2}:\d{2}(?::\d{2})?)\.?\s*$", line)
        if m or tail:
            ts, title = (m[1], m[2]) if m else (tail[2], tail[1])
            # если таймкод повторяется, оставляем первое название
            out.setdefault(parse_timestamp(ts), title.strip(" .-"))
    return [{"start": t, "title": out[t]} for t in sorted(out)]


def normalize_transcript(value):
    # приводим расшифровку к одному виду: список сегментов с началом, концом и текстом
    if isinstance(value, list):
        value = {"segments": value}
    segs, duration = [], float(value.get("duration", 0))
    for raw in value.get("segments", value.get("turns", [])):
        a, b = float(raw["start"]), float(raw["end"])
        text = str(raw.get("text", "")).strip()
        # сегмент с отрицательным или нулевым временем нам не нужен, лучше упасть сразу
        if not math.isfinite(a + b) or a < 0 or b <= a:
            raise ValueError(f"Некорректные времена сегмента: {a}, {b}")
        duration = max(duration, b)
        # пустые сегменты пропускаем, они ничего не дают сегментации
        if text:
            segs.append({"start": a, "end": b, "text": text, "speaker": raw.get("speaker")})
    if not segs or duration <= 0:
        raise ValueError("Нет распознанной речи. Проверьте источник и язык аудио.")
    return {**value, "duration": duration, "segments": sorted(segs, key=lambda x: (x["start"], x["end"]))}


def make_blocks(transcript, block_sec=30.0):
    # режем запись на блоки фиксированной длины (по умолчанию 30 секунд)
    # каждый сегмент расшифровки относим к блоку, в который попадает его середина
    duration = transcript["duration"]
    n = max(1, int(math.ceil(duration / block_sec)))
    groups = [[] for _ in range(n)]
    for s in transcript["segments"]:
        i = min(n - 1, int((s["start"] + s["end"]) / 2 / block_sec))
        groups[i].append(s)
    return [{"start": i * block_sec, "end": min(duration, (i + 1) * block_sec),
             "text": " ".join(s["text"] for s in g)} for i, g in enumerate(groups)]


@dataclass(frozen=True)
class SegmentConfig:
    # параметры сегментатора, которые мы подбираем на dev
    block_sec: float = 30.0   # длина одного блока текста
    window: int = 3           # сколько блоков берём слева и справа от стыка
    target_sec: float = 600.0 # желаемая средняя длина главы, из неё считаем число глав
    min_sec: float = 90.0     # короче полутора минут главу не делаем
    smooth: int = 3           # ширина сглаживания кривой близости


def depth_scores(similarity):
    # глубина провала как в texttiling: насколько точка ниже ближайших вершин слева и справа
    x = np.asarray(similarity, dtype=float)
    d = np.zeros(len(x))
    for i in range(len(x)):
        left, right = i, i
        # поднимаемся влево, пока близость растёт
        while left > 0 and x[left - 1] >= x[left]:
            left -= 1
        # и так же вправо
        while right + 1 < len(x) and x[right + 1] >= x[right]:
            right += 1
        d[i] = x[left] + x[right] - 2 * x[i]
    return d


def select_boundaries(times, scores, duration, count, min_sec):
    # выбираем count границ с максимальной суммарной глубиной так,
    # чтобы между соседними границами было не меньше min_sec. это динамическое программирование за O(kn)
    times, scores = np.asarray(times, float), np.asarray(scores, float)
    # кандидаты у самого начала и конца записи сразу отбрасываем
    ok = (times >= min_sec) & (times <= duration - min_sec)
    t, s = times[ok], scores[ok]
    # больше границ, чем помещается по минимальной длине, поставить нельзя
    count = min(int(count), max(0, int(duration // min_sec) - 1), len(t))
    if not count:
        return []
    # prev[i] - последний кандидат, который стоит не ближе min_sec до кандидата i
    prev = np.searchsorted(t, t - min_sec + 1e-9, side="right") - 1
    dp = np.full((count + 1, len(t)), -np.inf)
    parents = np.full(dp.shape, -1, int)
    # одна граница: её вес просто равен глубине
    dp[1] = s
    feasible = 1
    for k in range(2, count + 1):
        # для каждой позиции запоминаем лучший результат с k-1 границами левее неё
        prefix, ix = np.maximum.accumulate(dp[k - 1]), np.zeros(len(t), int)
        best = 0
        for i in range(len(t)):
            if dp[k - 1, i] > dp[k - 1, best]:
                best = i
            ix[i] = best
        for i, p in enumerate(prev):
            if p >= 0 and np.isfinite(prefix[p]):
                dp[k, i], parents[k, i] = s[i] + prefix[p], ix[p]
        # если k границ вообще можно разместить, запоминаем это
        if np.isfinite(dp[k]).any():
            feasible = k
    # восстанавливаем выбранные границы с конца
    i = int(np.argmax(dp[feasible]))
    chosen = []
    for k in range(feasible, 0, -1):
        chosen.append(float(t[i]))
        i = parents[k, i]
    return sorted(chosen)


def analyze_embeddings(blocks, embeddings, duration, config=SegmentConfig()):
    # главный шаг сегментации: из эмбеддингов блоков получаем границы глав
    e = np.asarray(embeddings, dtype=float)
    if len(e) != len(blocks):
        raise ValueError("Число эмбеддингов не совпало с числом блоков")
    if len(blocks) < 2:
        return {"bounds": [], "times": [], "similarity": [], "depth": []}
    # на каждом стыке сравниваем среднее окно слева со средним окном справа (косинусная близость)
    sim = []
    for i in range(1, len(e)):
        a, b = e[max(0, i - config.window):i].mean(0), e[i:i + config.window].mean(0)
        sim.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    sim = np.asarray(sim)
    # сглаживаем кривую, чтобы случайные провалы в одном блоке не становились главами
    width = min(config.smooth, len(sim))
    if width > 1:
        # окно делаем нечётным, тогда сглаживание не сдвигает кривую
        if width % 2 == 0:
            width -= 1
        sim = np.convolve(np.pad(sim, (width // 2, width // 2), mode="edge"), np.ones(width) / width, "valid")
    depth = depth_scores(sim)
    times = np.asarray([b["start"] for b in blocks[1:]])
    # число глав считаем из длительности и желаемой длины главы
    n_chapters = max(1, int(round(duration / config.target_sec)))
    # youtube показывает главы только если их минимум три, поэтому при достаточной длине записи берём не меньше трёх
    if duration >= 3 * config.min_sec:
        n_chapters = max(3, n_chapters)
    bounds = select_boundaries(times, depth, duration, n_chapters - 1, config.min_sec)
    return {"bounds": bounds, "times": times.tolist(), "similarity": sim.tolist(), "depth": depth.tolist()}


def uniform_boundaries(duration, target_sec):
    # baseline: просто режем запись на равные куски
    k = max(3, int(round(duration / target_sec)))
    return [float(x) for x in np.linspace(0, duration, k + 1)[1:-1]]


def random_boundaries(duration, target_sec, min_sec, seed=SEED):
    # baseline: случайные границы, но с той же минимальной длиной главы, что и у модели
    k = min(max(3, int(round(duration / target_sec))), max(1, int(duration // min_sec)))
    if k < 2:
        return []
    rng = np.random.default_rng(seed)
    # каждой главе даём минимальную длину, а остаток времени делим случайно
    lengths = min_sec + rng.dirichlet(np.ones(k)) * max(0.0, duration - k * min_sec)
    return np.cumsum(lengths)[:-1].tolist()


def match_boundaries(predicted, reference, tolerance=30):
    # сопоставляем предсказанные и авторские границы один к одному
    # динамикой максимизируем число пар, а при равенстве минимизируем суммарное смещение
    p, r = sorted(set(map(float, predicted))), sorted(set(map(float, reference)))
    n, m = len(p), len(r)
    tp, distance = np.zeros((n + 1, m + 1), int), np.zeros((n + 1, m + 1), float)
    moves = np.zeros((n + 1, m + 1), int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # варианты: пропустить предсказание, пропустить эталон или связать их в пару
            choices = [(tp[i - 1, j], distance[i - 1, j], 1), (tp[i, j - 1], distance[i, j - 1], 2)]
            if abs(p[i - 1] - r[j - 1]) <= tolerance:
                choices.append((tp[i - 1, j - 1] + 1, distance[i - 1, j - 1] + abs(p[i - 1] - r[j - 1]), 3))
            best = min(choices, key=lambda x: (-x[0], x[1]))
            tp[i, j], distance[i, j], moves[i, j] = best
    # проходим таблицу назад и собираем найденные пары
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        move = moves[i, j]
        if move == 3:
            pairs.append((i - 1, j - 1, abs(p[i - 1] - r[j - 1])))
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def time_window_metrics(predicted, reference, duration, grid_sec=10.0):
    # считаем pk и windowdiff на общей сетке с шагом 10 секунд
    # ширина окна k зависит только от эталона, так что все методы сравниваются честно
    positions = np.arange(0, duration, grid_sec)
    if len(positions) < 2:
        return float("nan"), float("nan")
    k = int(round(duration / (len(reference) + 1) / 2 / grid_sec))
    k = min(len(positions) - 1, max(1, k))
    # для каждой точки сетки считаем, сколько границ уже было до неё
    p, r = np.searchsorted(sorted(predicted), positions, side="right"), np.searchsorted(sorted(reference), positions, side="right")
    pdiff, rdiff = p[k:] - p[:-k], r[k:] - r[:-k]
    # pk: ошибка, если в окне есть граница только у одного из двух; windowdiff: если не совпадает число границ
    return float(np.mean((pdiff == 0) != (rdiff == 0))), float(np.mean(pdiff != rdiff))


def score_boundaries(predicted, reference, duration):
    # все метрики сегментации для одного выпуска
    # граница в 00:00 и конец записи не считаются, у них нет смысла как у найденной границы
    p = sorted({float(t) for t in predicted if 0 < t < duration})
    r = sorted({float(t) for t in reference if 0 < t < duration})
    row = {"n_pred": len(p), "n_gold": len(r), "duration": duration}
    # считаем с тремя допусками: 15, 30 и 60 секунд
    for tol in (15, 30, 60):
        pairs = match_boundaries(p, r, tol)
        tp, fp, fn = len(pairs), len(p) - len(pairs), len(r) - len(pairs)
        precision = tp / len(p) if p else (1.0 if not r else 0.0)
        recall = tp / len(r) if r else (1.0 if not p else 0.0)
        f1 = 2 * tp / (len(p) + len(r)) if p or r else 1.0
        row.update({f"P@{tol}": precision, f"R@{tol}": recall, f"F1@{tol}": f1,
                    f"TP@{tol}": tp, f"FP@{tol}": fp, f"FN@{tol}": fn,
                    f"MAE@{tol}": float(np.mean([a[2] for a in pairs])) if pairs else float("nan")})
    row["Pk"], row["WindowDiff"] = time_window_metrics(p, r, duration)
    return row


def rouge_l_f1(reference, candidate):
    # ROUGE-L F1 по Unicode-словам; правило токенизации одинаково для всех методов.
    # Регистр и различие ё/е снимаются, порядок слов сохраняется.
    def tokens(value):
        return re.findall(r"\w+", str(value).lower().replace("ё", "е"), flags=re.UNICODE)
    a, b = tokens(reference), tokens(candidate)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if word == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def bootstrap_mean(values, seed=SEED, n_boot=2000):
    # среднее и 95% интервал бутстрепом; пересэмплируем выпуски, а не отдельные границы
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (float("nan"),) * 3
    # по одному значению интервал не построить
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(v, (n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def chapter_segments(segments, start, end):
    # сегменты, середина которых попадает в главу
    return [s for s in segments if start <= (s["start"] + s["end"]) / 2 < end]


def excerpt(segments, max_words=750):
    # нам надо уложить главу в контекст llm, поэтому берём начало, середину и конец главы
    words = " ".join(s["text"] for s in segments).split()
    if len(words) <= max_words:
        return " ".join(words)
    n, mid = max_words // 3, len(words) // 2
    return " ".join(words[:n] + ["..."] + words[mid - n // 2:mid + n // 2] + ["..."] + words[-n:])


def speaker_breakdown(segments, turns, start, end, names=None):
    # доли спикеров в главе считаем по интервалам диаризации, а не по длине сегментов asr
    totals = {}
    for t in turns:
        overlap = max(0.0, min(end, t["end"]) - max(start, t["start"]))
        if overlap:
            totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + overlap
    total = sum(totals.values())
    # если продюсер задал имена, подставляем их вместо speaker_0, speaker_1
    return [{"id": k, "name": (names or {}).get(k, k), "share": v / total} for k, v in sorted(totals.items(), key=lambda x: -x[1])] if total else []


def chapters_from_bounds(transcript, bounds):
    # из границ собираем список глав с текстом для генерации названий
    starts = [0.0] + sorted({float(t) for t in bounds if 0 < t < transcript["duration"]})
    ends = starts[1:] + [transcript["duration"]]
    return [{"index": i, "start": a, "end": b, "text": excerpt(chapter_segments(transcript["segments"], a, b))}
            for i, (a, b) in enumerate(zip(starts, ends))]


def validate_youtube(chapters, duration):
    # проверяем правила youtube для глав: первый таймкод 00:00, минимум три главы, каждая не короче 10 секунд
    errors = []
    if len(chapters) < 3:
        errors.append("Для глав YouTube нужны минимум три таймкода")
    if not chapters or int(chapters[0]["start"]) != 0:
        errors.append("Первый таймкод должен быть 00:00")
    # длину сравниваем после округления до секунд, как её увидит youtube
    times = [int(c["start"]) for c in chapters] + [int(duration)]
    if any(b - a < 10 for a, b in zip(times, times[1:])):
        errors.append("Длина главы после округления таймкодов меньше 10 секунд")
    if any(not str(c.get("title", "")).strip() for c in chapters):
        errors.append("Есть пустое название")
    return errors


def make_description(chapters, duration, source="", show_name="Выпуск"):
    # собираем описание выпуска по шаблону: вводная строка, список таймкодов и источник
    # второй раз llm не зовём, чтобы не добавлять новых фактов
    errors = validate_youtube(chapters, duration)
    if errors:
        raise ValueError("; ".join(errors))
    intro = f"{show_name}. Обсуждаем: " + "; ".join(c["title"] for c in chapters[:6]) + "."
    lines = [intro, "", "Темы и таймкоды:"]
    lines += [f"{timestamp(c['start'])} {c['title']}" for c in chapters]
    if source:
        lines += ["", f"Источник: {source}"]
    description = "\n".join(lines)
    # у youtube лимит описания 5000 символов
    if len(description) > 5000:
        raise ValueError("Описание превышает 5000 символов. Сократите заголовки или число глав.")
    return description


def export_episode(directory, chapters, transcript, source="", show_name="Выпуск"):
    # сохраняем всё, что нужно продюсеру: описание, главы в json, расшифровку и главы в формате vtt
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    desc = make_description(chapters, transcript["duration"], source, show_name)
    (out / "description_youtube.txt").write_text(desc, encoding="utf-8")
    save_json(out / "chapters.json", chapters)
    save_json(out / "transcript.json", transcript)
    def vtt_time(t):
        # vtt требует время с миллисекундами
        n = int(round(t * 1000)); h, n = divmod(n, 3600000); m, n = divmod(n, 60000); s, ms = divmod(n, 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    lines = ["WEBVTT", ""]
    for c in chapters:
        lines += [f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}", c["title"], ""]
    (out / "chapters.vtt").write_text("\n".join(lines), encoding="utf-8")
    return desc


@dataclass(frozen=True)
class Economics:
    # все входы экономической модели в одном месте; это допущения, их мы уточняем на пилоте
    episodes_month: float = 32.5       # 7.5 выпуска в неделю * 52 / 12, середина диапазона из брифа
    manual_min: float = 75.0           # минут ручной работы на часовой выпуск
    loaded_hour_rub: float = 1000.0    # полная стоимость часа редактора, налоги уже внутри
    work_hours_month: float = 160.0    # рабочих часов в одном fte за месяц
    check_fixed_min: float = 8.0       # фиксированная проверка черновика на выпуск
    check_chapter_min: float = 0.75    # проверка одной предложенной главы
    remove_fp_min: float = 0.5         # удалить лишнюю границу
    find_fn_min: float = 10.0          # найти пропущенную границу: надо слушать запись, примерно как вручную (75 минут на 7 глав)
    title_edit_min: float = 1.5        # переписать одно название
    gpu_hour_rub: float = 150.0        # сценарная цена часа gpu, а не котировка поставщика
    support_hours_month: float = 3.0   # поддержка в месяц
    developer_hour_rub: float = 1500.0 # час разработчика
    setup_hours: float = 80.0          # внедрение
    horizon_months: int = 24           # горизонт расчёта roi


def economics_from_counts(tp, fp, fn, acceptance=0.5, rtf=0.1, config=Economics(), chapters_pred=None):
    # переводим ошибки границ (на час аудио) в минуты продюсера, fte и рубли
    if min(tp, fp, fn, rtf) < 0 or not 0 <= acceptance <= 1:
        raise ValueError("Неверные входы экономики")
    # tp, fp и fn считаются по внутренним границам, поэтому глав на одну больше: первая начинается в 00:00
    proposed = float(chapters_pred if chapters_pred is not None else tp + fp + 1)
    t = (config.check_fixed_min + proposed * config.check_chapter_min + fp * config.remove_fp_min
         + fn * config.find_fn_min + proposed * (1 - acceptance) * config.title_edit_min)
    # время t мы не обрезаем ручным процессом: плохой черновик может отнять больше времени, чем работа с нуля
    hours_before = config.episodes_month * config.manual_min / 60
    hours_after = config.episodes_month * t / 60
    hours_saved = hours_before - hours_after
    gross = hours_saved * config.loaded_hour_rub
    # вычисления: выпуск длится час, значит gpu-часов на выпуск ровно rtf
    compute = config.episodes_month * rtf * config.gpu_hour_rub
    support = config.support_hours_month * config.developer_hour_rub
    opex = compute + support
    capex = config.setup_hours * config.developer_hour_rub
    net = gross - opex
    horizon = config.horizon_months
    return {"minutes_after": t, "minutes_saved": config.manual_min - t,
            "hours_before": hours_before, "hours_after": hours_after, "hours_saved": hours_saved,
            "fte_before": hours_before / config.work_hours_month,
            "fte_after": hours_after / config.work_hours_month, "fte_saved": hours_saved / config.work_hours_month,
            "gross_month_rub": gross, "compute_month_rub": compute, "opex_month_rub": opex,
            "net_month_rub": net, "capex_rub": capex,
            "payback_months": capex / net if net > 0 else float("inf"),
            "roi": (horizon * gross - capex - horizon * opex) / (capex + horizon * opex)}


def economics_from_pr(precision, recall, n_gold=7, **kwargs):
    # сценарный вариант: из p и r восстанавливаем tp, fp и fn при заданном числе эталонных границ
    if not 0 < precision <= 1 or not 0 <= recall <= 1:
        raise ValueError("Сценарная формула требует 0 < precision <= 1 и 0 <= recall <= 1")
    tp = recall * n_gold
    return economics_from_counts(tp, tp / precision - tp, n_gold - tp, **kwargs)


def corrupt_segments(segments, probability, seed=SEED):
    # стресс-тест: портим текст asr удалениями и заменами слов
    # вероятность искажения здесь не равна wer реальной модели, это искусственная процедура
    rng = np.random.default_rng(seed)
    vocab = [w for s in segments for w in s["text"].split()]
    out = []
    for s in segments:
        words = []
        for word in s["text"].split():
            u = rng.random()
            # треть искажений делаем удалениями
            if u < probability / 3:
                continue
            # остальные заменяем случайным словом из того же выпуска
            words.append(vocab[int(rng.integers(len(vocab)))] if u < probability else word)
        out.append({**s, "text": " ".join(words)})
    return out


def word_errors(reference, hypothesis):
    # считаем число правок между двумя текстами (расстояние левенштейна по словам)
    def normalize(t):
        return re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е")).split()
    a, b = normalize(reference), normalize(hypothesis)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(row[j] + 1, cur[j - 1] + 1, row[j - 1] + (x != y)))
        row = cur
    return row[-1], len(a)


def word_error_rate(reference, hypothesis):
    # wer одной пары текстов
    edits, n = word_errors(reference, hypothesis)
    return edits / n if n else (0.0 if not edits else float("inf"))


def corpus_word_error_rate(references, hypotheses):
    # wer по корпусу: считаем правки по сегментам и делим на общее число слов
    # на целом выпуске сравнивать тексты целиком нельзя: это квадратичное время и часы ожидания
    total_edits, total_words = 0, 0
    for a, b in zip(references, hypotheses):
        edits, n = word_errors(a, b)
        total_edits, total_words = total_edits + edits, total_words + n
    return total_edits / total_words if total_words else 0.0
