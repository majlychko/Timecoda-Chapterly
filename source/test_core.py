import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from podcast_core import *
from podcast_models import extractive_title, streaming_inputs


def fixture():
    # искусственная расшифровка из трёх тем: так мы проверяем код, а не качество моделей на подкастах
    themes = ["футбол тренер команда стадион гол", "сервер код программа база данные", "кофе чашка кухня чай еда"]
    return {"duration": 900.0, "segments": [
        {"start": float(i * 30), "end": float((i + 1) * 30), "text": themes[i // 10], "speaker": None}
        for i in range(30)]}


class CoreTests(unittest.TestCase):
    # проверки чистых функций: таймкоды, разбор глав, сопоставление границ, метрики и экономика
    def test_timestamp(self):
        self.assertEqual(timestamp(3661.8), "01:01:01")

    def test_parse_timestamp(self):
        self.assertEqual(parse_timestamp("01:02:03"), 3723)

    def test_reject_bad_timestamp(self):
        with self.assertRaises(ValueError):
            parse_timestamp("02:99")

    def test_chapters_html(self):
        self.assertEqual(parse_chapters("00:00 Intro<br>02:00 <a href='x'>Тема</a>")[1]["start"], 120)

    def test_chapters_tail(self):
        self.assertEqual(parse_chapters("Тема \u2014 01:02:03.")[0]["start"], 3723)

    def test_chapters_duplicate(self):
        self.assertEqual(len(parse_chapters("00:00 Intro<br>00:00 Other")), 1)

    def test_transcript_invalid(self):
        with self.assertRaises(ValueError):
            normalize_transcript({"segments": [{"start": 20, "end": 10, "text": "x"}]})

    def test_blocks_keep_empty_intervals(self):
        tr = {"duration": 120, "segments": [{"start": 90, "end": 110, "text": "поздняя речь"}]}
        blocks = make_blocks(tr, 30)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0]["text"], "")

    def test_blocks_no_gold(self):
        self.assertEqual(make_blocks(fixture())[10]["start"], 300)

    def test_optimal_matching_beats_nearest_greedy(self):
        self.assertEqual(len(match_boundaries([0, 5], [4, 9], 4)), 2)

    def test_matching_is_one_to_one(self):
        self.assertEqual(len(match_boundaries([100, 101, 102], [100], 30)), 1)

    def test_matching_empty(self):
        self.assertEqual(match_boundaries([], [100]), [])

    def test_f1_perfect(self):
        self.assertEqual(score_boundaries([300, 600], [300, 600], 900)["F1@30"], 1)

    def test_f1_zero(self):
        self.assertEqual(score_boundaries([100], [500], 900)["F1@30"], 0)

    def test_f1_tolerance(self):
        r = score_boundaries([329], [300], 900)
        self.assertEqual(r["F1@15"], 0)
        self.assertEqual(r["F1@30"], 1)

    def test_start_and_end_excluded(self):
        self.assertEqual(score_boundaries([0, 300, 900], [300], 900)["n_pred"], 1)

    def test_pk_windowdiff_perfect(self):
        self.assertEqual(time_window_metrics([300, 600], [300, 600], 900), (0, 0))

    def test_window_metrics_miss(self):
        a, b = time_window_metrics([], [300, 600], 900)
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)

    def test_random_seed(self):
        self.assertEqual(random_boundaries(1800, 300, 90), random_boundaries(1800, 300, 90))

    def test_random_min_duration(self):
        b = [0] + random_boundaries(1800, 300, 90) + [1800]
        self.assertTrue(all(y - x >= 90 - 1e-8 for x, y in zip(b, b[1:])))

    def test_dp_min_duration(self):
        b = [0] + select_boundaries([100, 150, 300, 500, 750], [5, 9, 4, 3, 2], 900, 3, 120) + [900]
        self.assertTrue(all(y - x >= 120 for x, y in zip(b, b[1:])))

    def test_semantic_pipeline_fixture(self):
        tr = fixture(); b = make_blocks(tr)
        e = np.repeat(np.eye(3), 10, axis=0)
        p = analyze_embeddings(b, e, 900, SegmentConfig(window=2, target_sec=300, min_sec=90, smooth=1))
        self.assertEqual(p["bounds"], [300, 600])

    def test_bootstrap_reproducible(self):
        self.assertEqual(bootstrap_mean([0.2, 0.4, 0.6]), bootstrap_mean([0.2, 0.4, 0.6]))

    def test_single_episode_has_no_interval(self):
        self.assertTrue(math.isnan(bootstrap_mean([0.4])[1]))

    def test_economics_exact_fp_formula(self):
        a = economics_from_pr(0.5, 0.8, 10)
        b = economics_from_counts(8, 8, 2)
        self.assertAlmostEqual(a["minutes_after"], b["minutes_after"])

    def test_fn_more_expensive_than_fp(self):
        self.assertGreater(economics_from_counts(5, 0, 1)["minutes_after"], economics_from_counts(5, 1, 0)["minutes_after"])

    def test_bad_model_can_lose_time(self):
        self.assertLess(economics_from_counts(0, 0, 30)["minutes_saved"], 0)

    def test_payback_infinite(self):
        self.assertTrue(math.isinf(economics_from_counts(0, 0, 30)["payback_months"]))

    def test_roi_identity(self):
        e, r = Economics(), economics_from_counts(5, 1, 2)
        self.assertAlmostEqual(r["roi"], (24 * r["gross_month_rub"] - r["capex_rub"] - 24 * r["opex_month_rub"]) / (r["capex_rub"] + 24 * r["opex_month_rub"]))

    def test_economics_invalid(self):
        with self.assertRaises(ValueError):
            economics_from_counts(5, -1, 0)

    def test_youtube_validation(self):
        ch = [{"start": 0, "title": "Первый"}, {"start": 10, "title": "Второй"}, {"start": 20, "title": "Третий"}]
        self.assertEqual(validate_youtube(ch, 30), [])

    def test_no_fake_youtube_minimum(self):
        self.assertTrue(validate_youtube([{"start": 0, "title": "Тема"}], 30))

    def test_youtube_after_rounding(self):
        ch = [{"start": 0, "title": "Первый"}, {"start": 10.9, "title": "Второй"}, {"start": 20.1, "title": "Третий"}]
        self.assertTrue(validate_youtube(ch, 29.9))

    def test_speaker_identity_not_guessed(self):
        t = [{"start": 0, "end": 30, "speaker": "speaker_0"}, {"start": 30, "end": 60, "speaker": "speaker_1"}]
        self.assertEqual(speaker_breakdown([], t, 0, 60)[0]["name"], "speaker_0")

    def test_speaker_overlap_clipped(self):
        t = [{"start": 0, "end": 100, "speaker": "a"}, {"start": 60, "end": 80, "speaker": "b"}]
        shares = speaker_breakdown([], t, 50, 100)
        self.assertAlmostEqual(shares[0]["share"], 50 / 70)

    def test_corruption_zero(self):
        self.assertEqual(corrupt_segments(fixture()["segments"], 0), fixture()["segments"])

    def test_wer_known(self):
        self.assertAlmostEqual(word_error_rate("один два три", "один пять три"), 1 / 3)

    def test_full_export(self):
        tr = fixture()
        ch = chapters_from_bounds(tr, [300, 600])
        for i, c in enumerate(ch):
            c["title"] = f"Тема {i}"
        with tempfile.TemporaryDirectory() as directory:
            description = export_episode(directory, ch, tr)
            self.assertIn("00:00 Тема 0", description)
            self.assertEqual(len(json.loads((Path(directory) / "chapters.json").read_text())), 3)
            self.assertTrue((Path(directory) / "chapters.vtt").read_text().startswith("WEBVTT"))

    def test_streaming_short_and_final(self):
        class Processor:
            num_samples_first_audio_chunk = 10
            num_samples_per_audio_chunk = 10
            num_mel_frames_per_step = 5
            def audio_chunk_start(self, mel): return mel
            def __call__(self, wav, **kwargs): return {"n": len(wav), **kwargs}
        for length in [3, 10, 15, 16, 20, 30, 31]:
            chunks = list(streaming_inputs(Processor(), np.zeros(length)))
            self.assertEqual(sum(c.get("is_last_audio_chunk", False) for c in chunks), 1)
            self.assertTrue(chunks[-1]["is_last_audio_chunk"])
            self.assertTrue(all(c["n"] > 0 for c in chunks))

class CorpusWerTests(unittest.TestCase):
    # wer по сегментам должен совпадать с wer по склеенному тексту, если сегменты выровнены
    def test_corpus_wer_matches_joined(self):
        ref = ["один два три", "четыре пять"]
        hyp = ["один пять три", "четыре"]
        self.assertAlmostEqual(corpus_word_error_rate(ref, hyp), 2 / 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
