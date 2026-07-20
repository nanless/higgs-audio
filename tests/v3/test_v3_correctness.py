# Copyright (c) 2025 Boson AI

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOT = ROOT / "higgs_audio_v3_text_generator"
for path in (ROOT, TEXT_ROOT):
    sys.path.insert(0, str(path))


class AsrCacheIdentityTests(unittest.TestCase):
    def test_cache_entry_is_invalidated_when_audio_changes(self):
        from eval_higgs_audio.asr_cache import get_cached_asr_text, set_cached_asr_text

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clone_0001.wav"
            wav.write_bytes(b"first audio")
            cache = {}
            set_cached_asr_text(cache, wav, "Chinese", "old transcript")
            self.assertEqual(get_cached_asr_text(cache, wav, "Chinese"), "old transcript")

            time.sleep(0.002)
            wav.write_bytes(b"different replacement audio")
            self.assertIsNone(get_cached_asr_text(cache, wav, "Chinese"))

    def test_legacy_path_only_entry_is_not_trusted(self):
        from eval_higgs_audio.asr_cache import get_cached_asr_text

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clone_0001.wav"
            wav.write_bytes(b"audio")
            cache = {str(wav): {"text": "legacy", "language": "Chinese"}}
            self.assertIsNone(get_cached_asr_text(cache, wav, "Chinese"))


class EvaluationGateTests(unittest.TestCase):
    def test_missing_required_eval_is_not_keep(self):
        from eval_higgs_audio.postprocess_common import classify

        self.assertEqual(classify(None, 0.9), "MISSING_EVAL")
        self.assertEqual(classify(0.0, None), "MISSING_EVAL")
        self.assertEqual(classify(0.0, 0.9), "KEEP")

    def test_sim_only_gate_can_explicitly_ignore_cer(self):
        from eval_higgs_audio.postprocess_common import classify

        self.assertEqual(classify(None, 0.9, require_cer=False), "KEEP")
        self.assertEqual(classify(None, None, require_cer=False), "MISSING_EVAL")

    def test_prune_cli_deletes_nothing_when_any_required_eval_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            speaker = root / "dataset" / "speaker"
            speaker.mkdir(parents=True)
            missing = speaker / "clone_0001.wav"
            bad = speaker / "clone_0002.wav"
            missing.write_bytes(b"missing-cer")
            bad.write_bytes(b"would-be-deleted")
            (speaker / "clone_0001.sim.json").write_text(
                json.dumps({"cloned_audio": str(missing), "similarity": 0.9}), encoding="utf-8"
            )
            (speaker / "clone_0002.sim.json").write_text(
                json.dumps({"cloned_audio": str(bad), "similarity": 0.1}), encoding="utf-8"
            )
            (speaker / "clone_0002.cer.json").write_text(
                json.dumps({"wav_path": str(bad), "manual_cer": 0.5}), encoding="utf-8"
            )
            cmd = [
                sys.executable,
                str(ROOT / "eval_higgs_audio" / "prune_and_copy.py"),
                "--out-dir",
                str(root),
                "--workers",
                "1",
                "--scan-workers",
                "1",
                "--eval-workers",
                "1",
            ]
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertTrue(missing.exists())
            self.assertTrue(bad.exists(), "fail-closed must happen before deleting any clone")
            report = json.loads((root / "missing_eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["count"], 1)


class TextGenerationTests(unittest.TestCase):
    def test_task_targets_sum_to_exact_total(self):
        from higgs_text_gen.config import GenConfig
        from higgs_text_gen.task_generator import generate_task_list

        tasks = generate_task_list(GenConfig(total_target=10, batch_size=8, seed=42))
        self.assertEqual([task["target_count"] for task in tasks], [8, 2])
        self.assertEqual(sum(task["target_count"] for task in tasks), 10)

    def test_stable_seed_is_process_independent(self):
        from higgs_text_gen.stable_random import stable_int

        self.assertEqual(stable_int("task", 7, "emotion"), stable_int("task", 7, "emotion"))
        self.assertEqual(stable_int("task", 7, "emotion"), 16643962295530430178)

    def test_checkpoint_round_trip_is_atomic(self):
        from higgs_text_gen.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.jsonl"
            items = [{"task_id": 0, "text": "hello", "_private": "drop"}]
            save_checkpoint(items, str(path))
            self.assertEqual(load_checkpoint(str(path)), [{"task_id": 0, "text": "hello"}])
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_stale_checkpoint_rows_do_not_fake_resume(self):
        from higgs_audio_v3_text_generator.run_batch_generation import _sanitize_checkpoint

        task = {"task_id": 0, "target_count": 1, "task_signature": "current"}
        rows = [
            {"task_id": 0, "task_signature": "old", "text": "stale"},
            {"task_id": 0, "task_signature": "current", "text": "fresh"},
            {"task_id": 0, "task_signature": "current", "text": "overflow"},
        ]
        self.assertEqual(_sanitize_checkpoint(rows, {0: task}), [rows[1]])

    def test_matching_legacy_checkpoint_row_is_migrated(self):
        from higgs_audio_v3_text_generator.run_batch_generation import _sanitize_checkpoint

        task = {
            "task_id": 0,
            "target_count": 1,
            "task_signature": "current",
            "subscene": "lesson",
            "emotion": "joy",
            "length_key": "short",
            "lang_key": "pure_en",
        }
        legacy = {
            "task_id": 0,
            "text": "A valid legacy row.",
            "subscene": "lesson",
            "emotion": "joy",
            "length_type": "short",
            "lang_type": "pure_en",
        }
        migrated = _sanitize_checkpoint([legacy], {0: task})
        self.assertEqual(len(migrated), 1)
        self.assertEqual(migrated[0]["task_signature"], "current")

    def test_partial_batches_are_retried_and_resume_rebuilds_output(self):
        request_count = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                nonlocal request_count
                request_count += 1
                unique_word = chr(ord("A") + request_count - 1)
                body = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps([{"text": f"Unique generated text {unique_word}."}])}}
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("local sandbox does not allow loopback sockets")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / "output.jsonl"
                checkpoint = Path(td) / "checkpoint.jsonl"
                cmd = [
                    sys.executable,
                    str(TEXT_ROOT / "run_batch_generation.py"),
                    "--total",
                    "3",
                    "--batch-size",
                    "2",
                    "--workers",
                    "1",
                    "--seed",
                    "0",
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                    "--resume",
                    "--no-postprocess",
                ]
                first = subprocess.run(cmd, text=True, capture_output=True, check=False)
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
                self.assertEqual(request_count, 3)

                output.unlink()
                second = subprocess.run(cmd, text=True, capture_output=True, check=False)
                self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
                self.assertEqual(request_count, 3, "resume should not call the LLM for complete tasks")
        finally:
            server.shutdown()
            server.server_close()


class QualityFilterTests(unittest.TestCase):
    def _filter(self, text: str, length_type: str = "short"):
        from higgs_text_gen.quality_filter import quality_filter

        return quality_filter([{"text": text, "length_type": length_type}])

    def test_conflicting_styles_are_rejected(self):
        self.assertEqual(self._filter("<|style:shouting|><|style:whispering|>HELLO EVERYONE!"), [])

    def test_malformed_tag_like_token_is_rejected(self):
        self.assertEqual(self._filter("<|emotion:joy-1|>This should not pass."), [])
        self.assertEqual(self._filter("<|emotion:joy This should not pass."), [])

    def test_conflicts_are_order_independent(self):
        self.assertEqual(
            self._filter("<|prosody:speed_fast|><|prosody:speed_very_slow|>This cannot be both."),
            [],
        )

    def test_bad_marker_uses_word_boundaries(self):
        self.assertEqual(len(self._filter("She studied hard and passed.")), 1)

    def test_english_onomatopoeia_is_case_insensitive(self):
        self.assertEqual(len(self._filter("<|sfx:laughter|>haha! That was funny.")), 1)

    def test_severe_length_mismatch_is_rejected(self):
        self.assertEqual(self._filter("x" * 100, length_type="ultra_short"), [])


if __name__ == "__main__":
    unittest.main()
