# весь эксперимент в одном классе: сначала мы выбираем метод только по dev, потом один раз оцениваем его на test
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from podcast_core import *
from podcast_models import *

# модели, которые используются в нескольких этапах сразу
# кандидаты распознавания: все прогоняем на dev, на test идёт только тот, с которым главы на dev получились лучше
# порядок важен: при равенстве выигрывает модель выше в списке (gigaam сделана под русский и уже проверена в дз1)
ASR_CANDIDATES = {
    "gigaam-v3": {"id": "ai-sage/GigaAM-v3"},  # ветка e2e_rnnt, ревизия закреплена в data/model_revisions.json
    "whisper-turbo": {"id": "openai/whisper-large-v3-turbo", "language": "russian", "half": True},
    "parakeet-v3": {"id": "nvidia/parakeet-tdt-0.6b-v3", "half": False},
    "qwen3-asr": {"id": "Qwen/Qwen3-ASR-1.7B-hf", "language": "ru"},
}
EVAL_MODEL = "deepvk/USER2-base"  # оценщик названий и рубрик

# список рубрик для тематической навигации (идея списка тем из дз Бровина)
TOPICS = ["искусственный интеллект", "программирование и разработка", "гаджеты и техника", "бизнес и управление",
          "культура и медиа", "карьера и образование", "здоровье и спорт", "повседневная жизнь и общение"]


class Experiment:
    def __init__(self, work, manifest, revisions, device="cuda", limit_min=0, quick=False):
        # запоминаем папку запуска, список выпусков и закреплённые ревизии моделей
        self.work, self.manifest, self.device, self.limit_min = Path(work), manifest, device, limit_min
        self.work.mkdir(parents=True, exist_ok=True)
        self.revisions = {x["id"]: x["sha"] for x in revisions}
        self.ids = {part: [x["id"] for x in manifest if x["split"] == part] for part in ("dev", "test")}
        # в быстром режиме оставляем только два test-выпуска
        if quick:
            self.ids["test"] = self.ids["test"][:2]
        # dev и test не должны пересекаться, иначе оценка будет завышена
        assert not set(self.ids["dev"]) & set(self.ids["test"])
        assert len(self.ids["dev"]) >= 2 and len(self.ids["test"]) >= 2
        self.episodes = {x["id"]: dict(x) for x in manifest if x["id"] in self.ids["dev"] + self.ids["test"]}
        self.embeddings, self.timings, self.loads = {}, [], []
        self.configs, self.generations = {}, {}
        # два кандидата-эмбеддера; третий кандидат, tf-idf, не требует нейросети
        self.model_ids = {"user2-base": "deepvk/USER2-base", "qwen-embedding": "Qwen/Qwen3-Embedding-0.6B"}
        # сразу сохраняем манифест эксперимента, чтобы было видно, на каких данных и моделях он шёл
        save_json(self.work / "experiment_manifest.json", {"version": VERSION, "episodes": list(self.episodes.values()),
                  "dev": self.ids["dev"], "test": self.ids["test"], "limit_min": limit_min,
                  "seed": SEED, "model_revisions": self.revisions})

    def path(self, episode_id):
        # своя папка кэша на каждый выпуск
        p = self.work / "cache" / str(episode_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def record(self, stage, episode, seconds, **extra):
        # журнал времени этапов, из него потом считаем rtf и стоимость вычислений
        self.timings.append({"stage": stage, "episode": episode, "seconds": float(seconds), **extra})

    def prepare_audio(self):
        # скачиваем выпуски, переводим в 16 кгц моно и готовим авторские главы как эталон
        rows = []
        for i, ep in self.episodes.items():
            print(f"Аудио {i}: загрузка и приведение к 16 кГц mono", flush=True)
            original = download_file(ep["audio_url"], self.path(i) / "source.mp3")
            wav, duration = decode_to_wav(original, self.path(i) / f"audio_{self.limit_min:g}min.wav", self.limit_min)
            ep["audio"], ep["duration"] = str(wav), duration
            # конец главы - это начало следующей или конец записи
            gold = [{"start": c["start"], "end": (ep["chapters"][j + 1]["start"] if j + 1 < len(ep["chapters"]) else duration),
                     "title": c["title"]} for j, c in enumerate(ep["chapters"]) if c["start"] < duration]
            # в быстром режиме запись обрезана, поэтому и главы обрезаем по её длине
            for c in gold:
                c["end"] = min(c["end"], duration)
            if len(gold) < 2:
                raise ValueError(f"В {i} меньше двух авторских глав в выбранном фрагменте; увеличьте EVAL_LIMIT_MIN")
            # граница в 00:00 есть всегда, поэтому в оценку она не идёт
            ep["gold"], ep["gold_bounds"] = gold, [c["start"] for c in gold if c["start"] > 0]
            rows.append({"episode": i, "split": ep["split"], "minutes": duration / 60,
                         "gold_chapters": len(gold), "source": ep["page"]})
        self.data_table = pd.DataFrame(rows)
        self.data_table.to_csv(self.work / "data.csv", index=False)
        return self.data_table

    def load_asr(self, name):
        # загружаем одного кандидата распознавания; gigaam собирается вручную, остальные через пайплайн transformers
        spec = ASR_CANDIDATES[name]
        mid = spec["id"]
        if name == "gigaam-v3":
            return GigaAMASR(mid, self.revisions[mid], self.device)
        if name == "qwen3-asr":
            return QwenASR(mid, self.revisions[mid], self.device, language=spec["language"])
        return PipelineASR(mid, self.revisions[mid], self.device, language=spec.get("language"), half=spec.get("half", True))

    def segment_grid(self):
        # ширина окна, целевая длина главы и сглаживание: 24 варианта на каждый метод
        return [SegmentConfig(window=w, target_sec=t, smooth=s) for w, t, s in
                itertools.product((2, 4, 6), (300.0, 600.0, 900.0, 1200.0), (1, 3))]

    def cached_encode(self, model, mid, episode, texts, stage):
        # эмбеддинги блоков с кэшем: ключ зависит от модели и самих текстов, поэтому повторно не считаем
        key = digest([VERSION, mid, self.revisions[mid], texts, 30])
        target = self.path(episode) / f"emb_{key}.npz"
        if target.exists():
            v = np.load(target)
            embeddings, seconds, cached = v["embeddings"], float(v["seconds"]), True
        else:
            started = time.perf_counter()
            embeddings = model.encode(texts)
            seconds, cached = time.perf_counter() - started, False
            np.savez_compressed(target, embeddings=embeddings, seconds=seconds)
        self.record(stage, episode, seconds, cached=cached)
        return embeddings

    def transcribe(self):
        # 1. распознаём dev-выпуски всеми кандидатами; модели грузим по одной и сразу освобождаем память
        self.asr_names = [n for n, spec in ASR_CANDIDATES.items() if spec["id"] in self.revisions]
        self.asr_dev = {}
        for name in self.asr_names:
            print(f"Загружаю кандидата распознавания {name}. Другие модели пока не занимают GPU.", flush=True)
            model = self.load_asr(name)
            self.loads.append({"stage": "asr:" + name, "seconds": model.load_seconds})
            for i in self.ids["dev"]:
                print(f"ASR {name} {i}", flush=True)
                tr = model.transcribe(self.episodes[i]["audio"], self.path(i))
                self.asr_dev[(name, i)] = tr
                self.record("asr:" + name, i, tr["inference_seconds"], cached=tr["cached"])
            del model
            release_gpu()
        # 2. выбираем модель распознавания по качеству глав на dev
        self.select_asr()
        # 3. test распознаём только выбранной моделью, test в выборе не участвовал
        name = self.asr_selected
        print(f"Распознаю test выбранной моделью {name}", flush=True)
        model = self.load_asr(name)
        self.loads.append({"stage": "asr", "seconds": model.load_seconds})
        for i in self.ids["test"]:
            print(f"ASR {i}", flush=True)
            tr = model.transcribe(self.episodes[i]["audio"], self.path(i))
            self.episodes[i]["transcript"] = tr
            self.record("asr", i, tr["inference_seconds"], cached=tr["cached"])
        del model
        release_gpu()
        for i in self.ids["dev"]:
            self.episodes[i]["transcript"] = self.asr_dev[(name, i)]
        # tf-idf обучаем только на dev-выпусках выбранной модели, чтобы test никак не влиял на настройку
        dev_texts = [b["text"] for i in self.ids["dev"] for b in make_blocks(self.episodes[i]["transcript"])]
        self.tfidf = self.make_tfidf(dev_texts)
        # возвращаем короткую сводку: сколько сегментов и слов распознали и с какой скоростью
        return pd.DataFrame([{"episode": i, "split": e["split"], "asr": name, "segments": len(e["transcript"]["segments"]),
                              "words": sum(len(s["text"].split()) for s in e["transcript"]["segments"]),
                              "rtf_asr": e["transcript"]["inference_seconds"] / e["duration"],
                              "from_cache": e["transcript"]["cached"]} for i, e in self.episodes.items()])

    def make_tfidf(self, texts):
        # один и тот же tf-idf для выбора распознавания и для основной сегментации
        return TfidfVectorizer(stop_words=sorted(STOP), max_features=6000, min_df=1,
                               token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z-]{2,}\b", sublinear_tf=True).fit(texts)

    def select_asr(self):
        # для каждой модели распознавания повторяем на dev весь подбор сегментации (3 метода × 24 варианта)
        # и берём лучший f1@30; модель распознавания выбираем по нему, как раньше выбирали эмбеддер
        blocks = {(n, i): make_blocks(self.asr_dev[(n, i)]) for n in self.asr_names for i in self.ids["dev"]}
        vectors = {}
        for n in self.asr_names:
            tfidf = self.make_tfidf([b["text"] for i in self.ids["dev"] for b in blocks[(n, i)]])
            for i in self.ids["dev"]:
                vectors[(n, i, "tfidf")] = tfidf.transform([b["text"] for b in blocks[(n, i)]]).toarray()
        for method, mid in self.model_ids.items():
            print(f"Эмбеддинги {method} для выбора распознавания", flush=True)
            model = Encoder(mid, self.revisions[mid], self.device)
            for n in self.asr_names:
                for i in self.ids["dev"]:
                    vectors[(n, i, method)] = self.cached_encode(model, mid, i, [b["text"] for b in blocks[(n, i)]],
                                                                 "asr_selection:embedding")
            del model
            release_gpu()
        rows = []
        for n in self.asr_names:
            best = (-1.0, None, None)
            for method in ["tfidf", *self.model_ids]:
                for cfg in self.segment_grid():
                    f1 = float(np.mean([score_boundaries(analyze_embeddings(blocks[(n, i)], vectors[(n, i, method)],
                                                                            self.episodes[i]["duration"], cfg)["bounds"],
                                                         self.episodes[i]["gold_bounds"], self.episodes[i]["duration"])["F1@30"]
                                        for i in self.ids["dev"]]))
                    if f1 > best[0]:
                        best = (f1, method, cfg)
            dev_seconds = sum(self.asr_dev[(n, i)]["inference_seconds"] for i in self.ids["dev"])
            dev_duration = sum(self.episodes[i]["duration"] for i in self.ids["dev"])
            segments = [s for i in self.ids["dev"] for s in self.asr_dev[(n, i)]["segments"]]
            first = self.asr_names[0]
            rows.append({"asr": n, "model": ASR_CANDIDATES[n]["id"], "dev_F1@30": best[0], "лучший метод": best[1],
                         "rtf_dev": dev_seconds / dev_duration, "слов на час": 3600 * sum(len(s["text"].split()) for s in segments) / dev_duration,
                         "доля кусков не на русском": non_russian_share(segments),
                         f"расхождение с {first}": float(np.mean([transcript_disagreement(self.asr_dev[(first, i)], self.asr_dev[(n, i)])
                                                                for i in self.ids["dev"]])) if n != first else 0.0})
        self.asr_table = pd.DataFrame(rows)
        # при равенстве f1 выигрывает модель выше в списке кандидатов
        self.asr_selected = max(self.asr_names, key=lambda n: self.asr_table.set_index("asr").loc[n, "dev_F1@30"])
        self.asr_table["выбрана на dev"] = self.asr_table.asr == self.asr_selected
        self.asr_table.to_csv(self.work / "asr_selection_dev.csv", index=False)
        save_json(self.work / "frozen_asr_selection.json", {"selected_on": "dev only", "asr": self.asr_selected,
                  "model": ASR_CANDIDATES[self.asr_selected]["id"], "dev_ids": self.ids["dev"]})
        return self.asr_table

    def diarize_test(self):
        # диаризацию запускаем только на test: на dev она не нужна, сегментатор спикеров не использует
        mid = "nvidia/Nemotron-3-Diarization"
        started = time.perf_counter()
        model = NemotronDiarizer(self.revisions[mid], self.device)
        self.loads.append({"stage": "diarization", "seconds": time.perf_counter() - started})
        rows = []
        for i in self.ids["test"]:
            ep = self.episodes[i]
            print(f"Потоковая диаризация {i}", flush=True)
            r = model.diarize(ep["audio"], self.path(i))
            ep["diarization"] = r
            # каждому сегменту расшифровки приписываем спикера, который дольше всех говорит в этом интервале
            for segment in ep["transcript"]["segments"]:
                overlaps = speaker_breakdown([], r["turns"], segment["start"], segment["end"])
                segment["speaker"] = overlaps[0]["id"] if overlaps else None
            self.record("diarization", i, r["inference_seconds"], cached=r["cached"])
            rows.append({"episode": i, "speakers_detected": r["speaker_count"],
                         "rtf_diarization": r["inference_seconds"] / ep["duration"], "from_cache": r["cached"]})
        del model
        release_gpu()
        return pd.DataFrame(rows)

    def make_embeddings(self):
        # считаем векторы блоков всеми тремя способами: tf-idf, user2 и qwen
        for i, ep in self.episodes.items():
            ep["blocks"] = make_blocks(ep["transcript"])
            started = time.perf_counter()
            self.embeddings[(i, "tfidf")] = self.tfidf.transform([b["text"] for b in ep["blocks"]]).toarray()
            self.record("embedding:tfidf", i, time.perf_counter() - started)
        # нейросетевые эмбеддеры грузим по одному, чтобы не переполнить память t4
        for name, mid in self.model_ids.items():
            started = time.perf_counter()
            print(f"Эмбеддинги {name}", flush=True)
            model = Encoder(mid, self.revisions[mid], self.device)
            self.loads.append({"stage": "embedding:" + name, "seconds": time.perf_counter() - started})
            for i, ep in self.episodes.items():
                # dev-векторы выбранной модели уже посчитаны при выборе распознавания и берутся из кэша
                self.embeddings[(i, name)] = self.cached_encode(model, mid, i, [b["text"] for b in ep["blocks"]],
                                                                "embedding:" + name)
            del model
            release_gpu()

    def tune(self):
        # перебираем ширину окна, целевую длину главы и сглаживание; 24 варианта на каждый метод
        grid = self.segment_grid()
        rows = []
        for name in ["tfidf", *self.model_ids]:
            for cfg in grid:
                vals = []
                for i in self.ids["dev"]:
                    ep = self.episodes[i]
                    pred = analyze_embeddings(ep["blocks"], self.embeddings[(i, name)], ep["duration"], cfg)["bounds"]
                    vals.append(score_boundaries(pred, ep["gold_bounds"], ep["duration"])["F1@30"])
                # качество варианта: средний f1@30 по dev-выпускам
                rows.append({"method": name, **asdict(cfg), "dev_F1@30": float(np.mean(vals))})
            # для каждого метода фиксируем его лучшие параметры
            best = max([r for r in rows if r["method"] == name], key=lambda r: r["dev_F1@30"])
            self.configs[name] = SegmentConfig(**{k: best[k] for k in asdict(SegmentConfig())})
        self.tuning_table = pd.DataFrame(rows).sort_values("dev_F1@30", ascending=False, kind="stable")
        # выбираем один метод для продукта; при равенстве выигрывает более дешёвый: tfidf, потом user2, потом qwen
        self.selected = max(["tfidf", *self.model_ids], key=lambda n: max(r["dev_F1@30"] for r in rows if r["method"] == n))
        # длину главы для равномерного baseline тоже берём с dev, число глав test ему не сообщаем
        baseline_grid = []
        for target in (300.0, 600.0, 900.0, 1200.0):
            f1 = [score_boundaries(uniform_boundaries(self.episodes[i]["duration"], target),
                                   self.episodes[i]["gold_bounds"], self.episodes[i]["duration"])["F1@30"] for i in self.ids["dev"]]
            baseline_grid.append((float(np.mean(f1)), target))
        self.baseline_target = max(baseline_grid, key=lambda x: x[0])[1]
        # фиксируем выбор в файле до того, как смотрим на test
        self.tuning_table.to_csv(self.work / "dev_tuning.csv", index=False)
        save_json(self.work / "frozen_selection.json", {"selected_on": "dev only", "method": self.selected,
                  "configs": {k: asdict(v) for k, v in self.configs.items()}, "baseline_target": self.baseline_target,
                  "dev_ids": self.ids["dev"], "test_ids": self.ids["test"]})
        return self.tuning_table.groupby("method", sort=False).head(1)

    def evaluate(self):
        # один раз прогоняем все методы и baseline на test и считаем метрики
        rows, self.predictions = [], {}
        for i in self.ids["test"]:
            ep = self.episodes[i]
            for name, cfg in self.configs.items():
                started = time.perf_counter()
                result = analyze_embeddings(ep["blocks"], self.embeddings[(i, name)], ep["duration"], cfg)
                self.record("segmentation:" + name, i, time.perf_counter() - started)
                self.predictions[(i, name)] = result
                rows.append({"episode": i, "method": name, **score_boundaries(result["bounds"], ep["gold_bounds"], ep["duration"])})
            # равномерная нарезка
            uniform = uniform_boundaries(ep["duration"], self.baseline_target)
            rows.append({"episode": i, "method": "uniform", **score_boundaries(uniform, ep["gold_bounds"], ep["duration"])})
            # случайные границы: усредняем 20 нарезок, чтобы одна удачная не исказила картину
            random_rows = [score_boundaries(random_boundaries(ep["duration"], self.baseline_target, 90, SEED + i + rep),
                                            ep["gold_bounds"], ep["duration"]) for rep in range(20)]
            rows.append({"episode": i, "method": "random_mean20",
                         **{k: float(np.nanmean([r[k] for r in random_rows])) if any(np.isfinite(r[k]) for r in random_rows) else float("nan")
                            for k in random_rows[0]}})
        self.per_episode = pd.DataFrame(rows)
        # сводка по методам: среднее по выпускам и бутстреп-интервал
        summary = []
        for name, group in self.per_episode.groupby("method", sort=False):
            row = {"method": name, "n_episodes": len(group), "selected_on_dev": name == self.selected}
            for col in ("F1@15", "F1@30", "F1@60", "P@30", "R@30", "Pk", "WindowDiff"):
                mean, lo, hi = bootstrap_mean(group[col])
                row.update({col: mean, col + "_lo": lo, col + "_hi": hi})
            summary.append(row)
        self.summary = pd.DataFrame(summary)
        # парная разница с равномерной нарезкой: сравниваем методы на одних и тех же выпусках
        selected = self.per_episode[self.per_episode.method == self.selected].set_index("episode")
        uniform = self.per_episode[self.per_episode.method == "uniform"].set_index("episode")
        self.paired_delta = bootstrap_mean(selected["F1@30"] - uniform["F1@30"])
        self.per_episode.to_csv(self.work / "test_per_episode.csv", index=False)
        self.summary.to_csv(self.work / "test_summary.csv", index=False)
        return self.summary

    def build_chapter_inputs(self):
        # готовим главы двух видов: по границам модели и по авторским границам
        # второй вариант нужен, чтобы оценить названия отдельно от ошибок границ
        self.chapter_sets = {}
        for i in self.ids["test"]:
            ep = self.episodes[i]
            self.chapter_sets[(i, "predicted")] = chapters_from_bounds(ep["transcript"], self.predictions[(i, self.selected)]["bounds"])
            self.chapter_sets[(i, "gold_intervals")] = chapters_from_bounds(ep["transcript"], ep["gold_bounds"])
        return self.chapter_sets

    def generate(self):
        # генерируем название, краткую суть и цитату-опору для каждой главы
        self.build_chapter_inputs()
        mid = "Qwen/Qwen3-4B-Instruct-2507"
        print("Загружаю Qwen3-4B-Instruct-2507; ASR, диаризация и эмбеддеры освобождены.", flush=True)
        started = time.perf_counter()
        model = LocalGenerator(self.revisions[mid], self.device)
        self.loads.append({"stage": "llm", "seconds": time.perf_counter() - started})
        for (i, mode), chapters in self.chapter_sets.items():
            production_seconds = 0.0
            for c in chapters:
                # версия промпта входит в ключ кэша: поменяли промпт, значит пересчитываем
                key = digest([VERSION, mid, self.revisions[mid], "prompt-v2", c["text"]])
                dest = self.path(i) / f"title_{key}.json"
                if dest.exists():
                    data = json.loads(dest.read_text())
                else:
                    started = time.perf_counter()
                    data = model.generate_chapter(c["text"], extractive_title(c["text"], self.tfidf))
                    data["inference_seconds"] = time.perf_counter() - started
                    save_json(dest, data)
                c.update(data)
                # доли спикеров в главе
                c["speakers"] = speaker_breakdown([], self.episodes[i]["diarization"]["turns"], c["start"], c["end"])
                production_seconds += data["inference_seconds"]
            # в стоимость продукта идёт только генерация по границам модели
            if mode == "predicted":
                self.record("llm", i, production_seconds)
            self.generations[(i, mode)] = chapters
            print(f"Названия {i}, {mode}: {len(chapters)}", flush=True)
        del model
        release_gpu()
        return self.generations

    def evaluate_titles_and_topics(self):
        # оцениваем названия автоматически: косинус к названию автора и узнаёт ли название свою тему среди тем выпуска
        # оценщик всегда user2, это приближённая метрика, а не одобрение редактора
        # префикс classification, потому что здесь мы сравниваем название с темой, а не ищем кластеры
        mid = EVAL_MODEL
        started = time.perf_counter()
        model = Encoder(mid, self.revisions[mid], self.device, prefix="classification: ")
        self.loads.append({"stage": "topics", "seconds": time.perf_counter() - started})
        # векторы рубрик для zero-shot тематизации
        topic_vectors = model.encode(["Тема разговора: " + t for t in TOPICS])
        rows, review = [], []
        for i in self.ids["test"]:
            ep = self.episodes[i]
            refs = [c["title"] for c in ep["gold"]]
            ref_vectors = model.encode(refs)
            for mode in ("predicted", "gold_intervals"):
                chapters = self.generations[(i, mode)]
                variants = {"llm": [c["title"] for c in chapters]}
                # на авторских интервалах сравниваем llm с двумя простыми baseline
                if mode == "gold_intervals":
                    variants.update({"first_words": [" ".join(c["text"].split()[:7]) or "Нет речи" for c in chapters],
                                     "tfidf_title": [extractive_title(c["text"], self.tfidf) for c in chapters]})
                for name, titles in variants.items():
                    vectors = model.encode(titles)
                    for j, c in enumerate(chapters):
                        # эталонное название берём у авторской главы, которая сильнее всего пересекается с нашей
                        overlaps = [max(0.0, min(c["end"], g["end"]) - max(c["start"], g["start"])) for g in ep["gold"]]
                        target = int(np.argmax(overlaps))
                        scores = vectors[j] @ ref_vectors.T
                        rows.append({"episode": i, "intervals": mode, "method": name, "chapter": j,
                                     "cosine_to_author": float(scores[target]), "retrieval_hit1": float(np.argmax(scores) == target),
                                     "reference": refs[target], "generated": titles[j]})
                # рубрика главы - ближайшая по смыслу тема из списка; заодно собираем шаблон ручной оценки
                if mode == "predicted":
                    started = time.perf_counter()
                    text_vectors = model.encode([c["text"] for c in chapters])
                    for j, c in enumerate(chapters):
                        c["topic"] = TOPICS[int(np.argmax(text_vectors[j] @ topic_vectors.T))]
                        review.append({"episode": i, "chapter": j, "start": c["start"], "end": c["end"],
                                       "title": c["title"], "summary": c["summary"], "evidence": c["evidence"],
                                       "generation_status": c["generation_status"], "transcript_excerpt": c["text"],
                                       "rater": "", "title_accepted": "", "summary_supported": "", "comment": ""})
                    self.record("topics", i, time.perf_counter() - started)
        del model
        release_gpu()
        self.title_eval = pd.DataFrame(rows)
        self.review = pd.DataFrame(review)
        self.title_eval.to_csv(self.work / "title_proxy_metrics.csv", index=False)
        # это пустой шаблон для редакторов, сами оценки лежат отдельно в human_review.csv
        review_path = self.work / "human_review_template.csv"
        self.review.to_csv(review_path, index=False, encoding="utf-8-sig")
        self.title_summary = self.title_eval.groupby(["intervals", "method"])[["cosine_to_author", "retrieval_hit1"]].mean()
        return self.title_summary

    def export(self):
        # сохраняем по каждому test-выпуску описание, главы, расшифровку и vtt, плюс журналы времени
        self.descriptions = {}
        for i in self.ids["test"]:
            ep = self.episodes[i]
            self.descriptions[i] = export_episode(self.work / "outputs" / str(i), self.generations[(i, "predicted")],
                                                   ep["transcript"], ep["page"], ep["title"])
        pd.DataFrame(self.timings).to_csv(self.work / "timings.csv", index=False)
        pd.DataFrame(self.loads).to_csv(self.work / "model_loads.csv", index=False)
        return self.descriptions

    def production_rtf(self):
        # rtf продукта: суммируем только этапы, которые будут работать у заказчика
        # подбор параметров, проигравшие методы и стресс-тест сюда не входят
        required = {"asr", "diarization", "embedding:" + self.selected, "segmentation:" + self.selected, "llm", "topics"}
        table = pd.DataFrame(self.timings)
        prod = table[table.episode.isin(self.ids["test"]) & table.stage.isin(required)]
        seconds = prod.seconds.sum()
        duration = sum(self.episodes[i]["duration"] for i in self.ids["test"])
        stages = prod.groupby("stage").seconds.sum() / duration
        return float(seconds / duration), stages

    def evaluate_economics(self, config=Economics(), acceptance=None):
        # переводим ошибки выбранного метода на test в минуты, fte и рубли
        group = self.per_episode[self.per_episode.method == self.selected]
        # все счётчики приводим к одному часу аудио, потому что выпуск заказчика длится час
        hours = group.duration.sum() / 3600
        counts = {key: float(group[col].sum() / hours) for key, col in [("tp", "TP@30"), ("fp", "FP@30"), ("fn", "FN@30")]}
        n_chapters_hour = float((group.n_pred.sum() + len(group)) / hours)
        rtf, _ = self.production_rtf()
        # время загрузки моделей тоже учитываем, раскладывая его на месячную партию выпусков
        loads = [x["seconds"] for x in self.loads if x["stage"] in {"asr", "diarization", "embedding:" + self.selected, "llm", "topics"}]
        cold_rtf = sum(loads) / max(config.episodes_month * 3600, 1)
        inputs = {**counts, "rtf": rtf + cold_rtf, "chapters_pred": n_chapters_hour}
        # пока нет ручной оценки, долю принятых названий задаём сценариями
        scenarios = []
        for a in (0.25, 0.5, 0.75):
            scenarios.append({"title_acceptance_ASSUMPTION": a, **economics_from_counts(**inputs, acceptance=a, config=config)})
        self.economics_table = pd.DataFrame(scenarios)
        self.economics_table.to_csv(self.work / "economics_scenarios.csv", index=False)
        result = {"status": "scenario_not_measured_saving", "title_acceptance_assumed": acceptance is None,
                  "title_acceptance": 0.5 if acceptance is None else acceptance,
                  "rtf_warm": rtf, "rtf_with_monthly_model_load": rtf + cold_rtf,
                  "counts_per_audio_hour": counts, "config": asdict(config),
                  **economics_from_counts(**inputs, acceptance=0.5 if acceptance is None else acceptance, config=config)}
        save_json(self.work / "economics.json", result)
        self.economics_result, self.economics_inputs = result, inputs
        return self.economics_table

    def stress_test(self):
        # проверяем, насколько главы устойчивы к ошибкам распознавания: портим текст и повторяем сегментацию
        name, cfg = self.selected, self.configs[self.selected]
        encoder = None
        if name in self.model_ids:
            mid = self.model_ids[name]
            encoder = Encoder(mid, self.revisions[mid], self.device)
        rows = []
        for probability in (0.0, 0.1, 0.3):
            for i in self.ids["test"]:
                ep, original = self.episodes[i], self.episodes[i]["transcript"]
                modified = {**original, "segments": corrupt_segments(original["segments"], probability, SEED + i)}
                blocks = make_blocks(modified)
                texts = [b["text"] for b in blocks]
                e = encoder.encode(texts) if encoder else self.tfidf.transform(texts).toarray()
                bounds = analyze_embeddings(blocks, e, ep["duration"], cfg)["bounds"]
                # wer между исходным и испорченным текстом считаем по сегментам, это быстро даже на двухчасовом выпуске
                added_wer = corpus_word_error_rate([s["text"] for s in original["segments"]],
                                                   [s["text"] for s in modified["segments"]])
                rows.append({"episode": i, "corruption_probability": probability, "added_synthetic_WER": added_wer,
                             **score_boundaries(bounds, ep["gold_bounds"], ep["duration"])})
        if encoder is not None:
            del encoder
            release_gpu()
        self.stress_table = pd.DataFrame(rows)
        self.stress_table.to_csv(self.work / "synthetic_stress_test.csv", index=False)
        return self.stress_table.groupby("corruption_probability")[["added_synthetic_WER", "F1@30"]].mean()

    def demo_new(self, source, limit_min=0, speaker_names=None):
        # полный сценарий продюсера на новой записи: тот же пайплайн и те же параметры, что выбраны на dev
        directory = self.work / "custom" / digest([source, limit_min])
        wav, duration = resolve_media(source, directory, limit_min)
        # распознавание
        asr = self.load_asr(self.asr_selected)
        transcript = asr.transcribe(wav, directory)
        del asr
        release_gpu()
        # диаризация
        diarizer = NemotronDiarizer(self.revisions["nvidia/Nemotron-3-Diarization"], self.device)
        diarization = diarizer.diarize(wav, directory)
        for segment in transcript["segments"]:
            labels = speaker_breakdown([], diarization["turns"], segment["start"], segment["end"])
            segment["speaker"] = labels[0]["id"] if labels else None
        del diarizer
        release_gpu()
        # сегментация выбранным на dev методом
        blocks = make_blocks(transcript)
        texts = [b["text"] for b in blocks]
        if self.selected == "tfidf":
            vectors = self.tfidf.transform(texts).toarray()
        else:
            mid = self.model_ids[self.selected]
            encoder = Encoder(mid, self.revisions[mid], self.device)
            vectors = encoder.encode(texts)
            del encoder
            release_gpu()
        result = analyze_embeddings(blocks, vectors, duration, self.configs[self.selected])
        chapters = chapters_from_bounds(transcript, result["bounds"])
        # названия и суть глав
        generator = LocalGenerator(self.revisions["Qwen/Qwen3-4B-Instruct-2507"], self.device)
        for c in chapters:
            c.update(generator.generate_chapter(c["text"], extractive_title(c["text"], self.tfidf)))
            c["speakers"] = speaker_breakdown([], diarization["turns"], c["start"], c["end"], speaker_names)
        del generator
        release_gpu()
        # рубрики
        mid = EVAL_MODEL
        encoder = Encoder(mid, self.revisions[mid], self.device, prefix="classification: ")
        topics = encoder.encode(["Тема разговора: " + t for t in TOPICS])
        vectors = encoder.encode([c["text"] for c in chapters])
        for c, vector in zip(chapters, vectors):
            c["topic"] = TOPICS[int(np.argmax(vector @ topics.T))]
        del encoder
        release_gpu()
        # сохраняем всё в папку нового выпуска
        description = export_episode(directory / "outputs", chapters, transcript,
                                     source if str(source).startswith("http") else "", "Новый выпуск")
        return chapters, description, directory / "outputs"
