# Gender / Pause Clone Demo

This untracked V3 experiment samples adult and child speakers, assigns a curated
multilingual corpus, clones with SGLang-Omni, and optionally applies VAD-based
pause resplicing and background-noise mixing.

## Inputs and provenance

`run_all.sh` defaults to internal dataset paths. Override `TRAIN_JSON`,
`GENDER_JSON`, and `AUDIO_ROOT` for another environment. References must live
under `AUDIO_ROOT` and have non-empty transcript sidecars.

`clone_text_corpus.json` is a curated data artifact, not generated at runtime.
Before committing or redistributing it, record its author, license, generation
script/version, and review status.

## Dependencies

- Python 3.10+
- `numpy`, `requests`, `silero-vad`
- optional `pyroomacoustics` for reverb
- an available Higgs Audio V3 SGLang-Omni `/v1/audio/speech` endpoint
- noise data under `NOISE_ROOT` for stage 05

## Pipeline

```bash
TRAIN_JSON=/path/train.json \
GENDER_JSON=/path/per_speaker.json \
AUDIO_ROOT=/path/audio \
BASE_URL=http://localhost:8000 \
bash run_all.sh
```

Stages 01–03 run by default. Optional postprocessing is explicit because stages
04 and 05 modify clone WAV/JSON files in place (stage 05 preserves
`clone_dry.wav`):

```bash
RESPLICE=1 MIX_BG_NOISE=1 bash run_all.sh
```

All random choices use SHA-256-derived stable seeds, so the same inputs and
`SEED` are reproducible across Python processes and hosts.
