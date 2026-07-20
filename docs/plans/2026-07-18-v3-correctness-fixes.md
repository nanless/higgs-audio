# Higgs Audio V3 Correctness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the confirmed V3 production correctness issues without touching V2 code or mutating existing clone artifacts.

**Architecture:** Preserve the current pipelines and file formats where possible, but add explicit identities and fail-closed gates at data boundaries. Use deterministic SHA-256-derived seeds, atomic state writes, exact per-task targets, and regression tests based on the Python standard library so they run in the repository's minimal environment.

**Tech Stack:** Python 3.10+, Bash, JSON/JSONL, unittest, existing Higgs V3 scripts.

---

### Task 1: Regression test foundation

**Files:**
- Create: `tests/v3/test_v3_correctness.py`

**Steps:**
1. Add failing tests for ASR cache fingerprint invalidation and missing-evaluation classification.
2. Add failing tests for exact task targets, atomic checkpoint recovery, stable seeds, and strict tag/length validation.
3. Run `python3 -m unittest discover -s tests/v3 -v` and confirm the expected failures.

### Task 2: Clone/evaluation identity and fail-closed gates

**Files:**
- Modify: `eval_higgs_audio/eval_cer/eval_cer.py`
- Create: `eval_higgs_audio/asr_cache.py`
- Modify: `eval_higgs_audio/postprocess_common.py`
- Modify: `eval_higgs_audio/prune_and_copy.py`
- Modify: `eval_higgs_audio/verify_kept_clones.py`
- Modify: `v3_tts_clone/05_iterative_pipeline.sh`

**Steps:**
1. Store WAV size, inode, and nanosecond mtime/ctime with every ASR cache entry; reject legacy or mismatched entries.
2. Classify missing required CER/SIM as `MISSING_EVAL`, never `KEEP`.
3. Abort prune before deletion when required evaluations are missing, and write missing paths to a report.
4. Make SIM/CER evaluation and prune failures stop the production round instead of continuing.
5. Run focused unit tests and shell syntax checks.

### Task 3: Resume budget and supervisor freshness

**Files:**
- Modify: `v3_tts_clone/07_topup_pipeline.sh`
- Modify: `v3_tts_clone/09_stop_after_round.sh`

**Steps:**
1. Restore `TOTAL_CLONE_HOURS` from the allocation summary during resume and reject missing/invalid state.
2. Require completion artifacts/logs to be newer than the supervisor start time.
3. Run `bash -n` and isolated temporary-directory shell tests.

### Task 4: Text generation correctness

**Files:**
- Create: `higgs_audio_v3_text_generator/higgs_text_gen/stable_random.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/task_generator.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/compact_prompt.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/worker.py`
- Modify: `higgs_audio_v3_text_generator/run_batch_generation.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/checkpoint.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/output.py`
- Modify: `higgs_audio_v3_text_generator/run_parallel_batch.py`
- Modify: `higgs_audio_v3_text_generator/run_1m_gen.sh`
- Modify: `higgs_audio_v3_text_generator/postprocess_merge.py`

**Steps:**
1. Replace built-in `hash()` seeds with stable SHA-256-derived integers.
2. Generate ceil-divided tasks with an exact `target_count` for the final partial task.
3. Bind checkpoints to deterministic task signatures, track accepted per-task counts, and resubmit partial tasks with bounded attempts.
4. Use atomic checkpoint replacement and preserve checkpoint/output files on restart.
5. Dynamically allocate instance totals/ports, propagate child failures, and stream the merge.
6. Invoke final postprocessing from the parallel entrypoint and fail clearly when the final accepted count is below target.
7. Run unit tests and small fake-LLM integration tests.

### Task 5: Strict text quality validation

**Files:**
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/quality_filter.py`
- Modify: `higgs_audio_v3_text_generator/higgs_text_gen/tag_guide.py`

**Steps:**
1. Reject malformed tag-like tokens, incompatible tag combinations, and severe length mismatch.
2. Make English bad-marker matching word-boundary aware and SFX matching case-insensitive.
3. Enforce shouting semantics when the generated text contains Latin letters.
4. Run focused regression tests.

### Task 6: Untracked demo reproducibility and documentation

**Files:**
- Create: `v3_tts_clone/demo_gender_pause_clone/README.md`
- Create: `v3_tts_clone/demo_gender_pause_clone/stable_seed.py`
- Modify: `v3_tts_clone/demo_gender_pause_clone/01_sample_speakers.py`
- Modify: `v3_tts_clone/demo_gender_pause_clone/03_run_clone.py`
- Modify: `v3_tts_clone/demo_gender_pause_clone/04_resplice_variable_pause.py`
- Modify: `v3_tts_clone/demo_gender_pause_clone/05_mix_bg_noise.py`
- Modify: `v3_tts_clone/demo_gender_pause_clone/run_all.sh`

**Steps:**
1. Replace all built-in hash-derived seeds.
2. Document internal-path parameters, dependencies, corpus provenance requirement, and stages 04/05.
3. Make optional resplice/noise stages available from `run_all.sh` without changing current defaults.
4. Exclude generated bytecode from the delivered file set.

### Task 7: Final verification and remote delivery

**Steps:**
1. Run all V3 unit tests, Python AST parsing, and `bash -n`.
2. Review the complete diff and confirm no `boson_multimodal/` path is touched.
3. Sync only reviewed source/test/plan files to `dev_a800_8gpus`.
4. Repeat tests on the remote host and report any environment-limited checks.
