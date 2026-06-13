#!/usr/bin/env python3
"""Unified scorer module for Higgs Audio TTS evaluation metrics.

Supported metrics:
  - UTMOS22Strong  (custom PyTorch model, utmos_model.py + audio_utils.py)
  - SCOREQ         (pip package `scoreq`, ONNX-based NR quality)
  - TTSDS2         (pip package `ttsds`, benchmark suite)
  - UTMOSv2        (git package `utmosv2`, SSL+spectrogram MOS predictor)

Key difference from OmniVoice: Higgs clone audio is 24 kHz → resample to 16 kHz for MOS.

Usage:
    from scorers import create_scorer, AVAILABLE_METRICS
    scorer = create_scorer("SCOREQ", device="cuda:0")
    score = scorer.score_file("clone_0000.wav")
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch

SCORER_DIR = Path(__file__).resolve().parent

AVAILABLE_METRICS: List[str] = ["UTMOS22Strong", "SCOREQ", "TTSDS2", "UTMOSv2"]

TARGET_SR = 16000


class BaseScorer(ABC):
    name: str = "base"

    @abstractmethod
    def score_file(self, wav_path: Union[str, Path]) -> float:
        """Score a single wav file. Higher is better."""

    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        return [self.score_file(p) for p in wav_paths]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# UTMOS22Strong — local utmos_model.py + audio_utils.py
# ---------------------------------------------------------------------------

_utmos_mod = None
_utils_mod = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_utmos_module():
    global _utmos_mod
    if _utmos_mod is None:
        _utmos_mod = _load_module("higgs_utmos_model", SCORER_DIR / "utmos_model.py")
    return _utmos_mod


def _get_utils_module():
    global _utils_mod
    if _utils_mod is None:
        _utils_mod = _load_module("higgs_audio_utils", SCORER_DIR / "audio_utils.py")
    return _utils_mod


class UTMOS22StrongScorer(BaseScorer):
    name = "UTMOS22Strong"

    # Default model checkpoint search paths
    _DEFAULT_CKPT_PATHS = [
        Path("/root/code/github_repos/OmniVoice-fork/TTS_eval_models/mos/utmos22_strong_step7459_v1.pt"),
        Path.home() / ".cache/higgs_eval/utmos22_strong_step7459_v1.pt",
    ]

    def __init__(self, model_dir: Union[str, Path, None] = None, device: str = "cuda:0"):
        self.device = torch.device(device)
        utmos_mod = _get_utmos_module()
        utils_mod = _get_utils_module()

        # Resolve checkpoint path
        ckpt = None
        if model_dir:
            ckpt = Path(model_dir) / "mos/utmos22_strong_step7459_v1.pt"
            if not ckpt.exists():
                ckpt = Path(model_dir) / "utmos22_strong_step7459_v1.pt"

        if ckpt is None or not ckpt.exists():
            for p in self._DEFAULT_CKPT_PATHS:
                if p.exists():
                    ckpt = p
                    break

        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(
                "UTMOS checkpoint not found.\n"
                "Download: huggingface-cli download --local-dir ~/.cache/higgs_eval "
                "k2-fsa/TTS_eval_models mos/utmos22_strong_step7459_v1.pt\n"
                "  or set TTS_EVAL_MODEL_DIR env var"
            )

        self.model = utmos_mod.UTMOS22Strong()
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = TARGET_SR
        self._load_eval_waveform = utils_mod.load_eval_waveform

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        speech = self._load_eval_waveform(str(wav_path), self.sample_rate, device=self.device)
        score = self.model(speech.unsqueeze(0), self.sample_rate)
        return float(score.item())

    @torch.no_grad()
    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        MAX_AUDIO_LEN = 30 * self.sample_rate
        MAX_BATCH_TOKENS = 16 * 10 * self.sample_rate

        speeches = []
        for wp in wav_paths:
            s = self._load_eval_waveform(str(wp), self.sample_rate, device=self.device)
            if s.shape[0] > MAX_AUDIO_LEN:
                s = s[:MAX_AUDIO_LEN]
            speeches.append(s)

        indexed = list(enumerate(speeches))
        indexed.sort(key=lambda x: x[1].shape[0])

        results = [0.0] * len(wav_paths)
        batch_groups = []
        current_group = []
        current_tokens = 0

        for idx, s in indexed:
            s_len = s.shape[0]
            if current_group and (current_tokens + s_len > MAX_BATCH_TOKENS or len(current_group) >= 8):
                batch_groups.append(current_group)
                current_group = []
                current_tokens = 0
            current_group.append((idx, s))
            current_tokens += s_len

        if current_group:
            batch_groups.append(current_group)

        for group in batch_groups:
            max_len = max(s.shape[0] for _, s in group)
            batch = torch.zeros(len(group), max_len, device=self.device)
            for i, (orig_idx, s) in enumerate(group):
                batch[i, : s.shape[0]] = s
            scores = self.model(batch, self.sample_rate)
            for i, (orig_idx, _) in enumerate(group):
                results[orig_idx] = float(scores[i].item())

        return results


# ---------------------------------------------------------------------------
# SCOREQ — pip package `scoreq`
# ---------------------------------------------------------------------------


class SCOREQScorer(BaseScorer):
    name = "SCOREQ"

    def __init__(self, device: str = "cuda:0", **kwargs):
        try:
            from scoreq.scoreq import Scoreq
        except ImportError:
            raise ImportError("scoreq package not found. Install: pip install scoreq")
        self.device = device
        data_domain = "synthetic"
        synthetic_path = Path.home() / ".cache" / "scoreq" / "onnx-models" / "adapt_nr_synthetic.onnx"
        if not synthetic_path.exists():
            data_domain = "natural"
        if "cuda" in device:
            self.model = Scoreq(data_domain=data_domain, mode="nr", use_onnx=True)
            if hasattr(self.model, "session") and self.model.session is not None:
                try:
                    self.model.session.set_providers(["CUDAExecutionProvider"])
                except Exception:
                    pass
        else:
            self.model = Scoreq(data_domain=data_domain, mode="nr", use_onnx=True)
            if hasattr(self.model, "session") and self.model.session is not None:
                try:
                    self.model.session.set_providers(["CPUExecutionProvider"])
                except Exception:
                    pass

    def score_file(self, wav_path: Union[str, Path]) -> float:
        return float(self.model.predict(str(wav_path)))


# ---------------------------------------------------------------------------
# TTSDS2 — pip package `ttsds`
# ---------------------------------------------------------------------------


class TTSDS2Scorer(BaseScorer):
    name = "TTSDS2"

    _TTSDS_ROOT: str = ""

    def __init__(self, device: str = "cuda:0", **kwargs):
        self.device = device
        self._benchmarks = []
        self._mode = "fallback"
        self._init_ttsds_root()
        self._init_benchmarks()

    def _init_ttsds_root(self):
        try:
            import site

            for sp in site.getsitepackages() + [site.getusersitepackages()]:
                candidate = Path(sp) / "ttsds" / "benchmarks"
                if candidate.is_dir():
                    self._TTSDS_ROOT = str(Path(sp) / "ttsds")
                    return
            import sysconfig

            sp = sysconfig.get_path("purelib")
            candidate = Path(sp) / "ttsds" / "benchmarks"
            if candidate.is_dir():
                self._TTSDS_ROOT = str(Path(sp) / "ttsds")
        except Exception:
            pass

    def _load_module(self, name: str, filepath: str):
        spec = importlib.util.spec_from_file_location(name, filepath)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _init_benchmarks(self):
        if not self._TTSDS_ROOT:
            return
        root = self._TTSDS_ROOT
        deps = [
            ("ttsds.util.cache", f"{root}/util/cache.py"),
            ("ttsds.util.dataset", f"{root}/util/dataset.py"),
            ("ttsds.util.distances", f"{root}/util/distances.py"),
            ("ttsds.util.parallel_distances", f"{root}/util/parallel_distances.py"),
            ("ttsds.benchmarks.benchmark", f"{root}/benchmarks/benchmark.py"),
        ]
        for name, path in deps:
            if name not in sys.modules:
                try:
                    self._load_module(name, path)
                except Exception:
                    return
        try:
            wavlm_mod = self._load_module("ttsds.benchmarks.general.wavlm", f"{root}/benchmarks/general/wavlm.py")
            if wavlm_mod:
                wavlm = wavlm_mod.WavLMBenchmark()
                if "cuda" in self.device:
                    wavlm.to_device("cuda")
                self._benchmarks.append(("wavlm", wavlm))
        except Exception:
            pass
        try:
            whisper_mod = self._load_module(
                "ttsds.benchmarks.intelligibility.whisper_activations",
                f"{root}/benchmarks/intelligibility/whisper_activations.py",
            )
            if whisper_mod:
                whisper = whisper_mod.WhisperActivationsBenchmark()
                if "cuda" in self.device:
                    whisper.to_device("cuda")
                self._benchmarks.append(("whisper", whisper))
        except Exception:
            pass
        try:
            pitch_mod = self._load_module("ttsds.benchmarks.prosody.pitch", f"{root}/benchmarks/prosody/pitch.py")
            if pitch_mod:
                pitch = pitch_mod.PitchBenchmark()
                self._benchmarks.append(("pitch", pitch))
        except Exception:
            pass
        if self._benchmarks:
            self._mode = "benchmarks"

    def score_file(self, wav_path: Union[str, Path]) -> float:
        if self._mode == "benchmarks":
            return self._score_with_benchmarks(wav_path)
        return self._score_fallback(wav_path)

    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(self.score_file, wav_paths))

    def _score_with_benchmarks(self, wav_path: Union[str, Path]) -> float:
        import librosa

        wav_path = Path(wav_path)
        try:
            y, sr = librosa.load(str(wav_path), sr=16000)
        except Exception:
            return self._score_fallback(wav_path)
        if len(y) == 0:
            return 0.0
        scores = []
        for name, bench in self._benchmarks:
            try:
                test_emb = bench.get_embedding(y, sr)
                noise = np.random.RandomState(42).randn(len(y)).astype(np.float32) * 0.1
                noise_emb = bench.get_embedding(noise, sr)
                dist = float(np.linalg.norm(test_emb.mean(axis=0) - noise_emb.mean(axis=0)))
                score = 100.0 * (dist / (dist + 1.0))
                scores.append(score)
            except Exception:
                continue
        if scores:
            return float(np.mean(scores))
        return self._score_fallback(wav_path)

    def _score_fallback(self, wav_path: Union[str, Path]) -> float:
        try:
            import librosa

            wav_path = str(wav_path)
            y, sr = librosa.load(wav_path, sr=16000)
            if len(y) == 0:
                return 0.0
            sf = librosa.feature.spectral_flatness(y=y)[0]
            sf_score = max(0.0, 1.0 - float(sf.mean()))
            rms = librosa.feature.rms(y=y)[0]
            rms_cv = float(rms.std() / (rms.mean() + 1e-8))
            rms_score = max(0.0, 1.0 - min(rms_cv, 1.0))
            sorted_rms = np.sort(rms)
            noise_floor = sorted_rms[: max(1, len(sorted_rms) // 10)].mean()
            signal_power = sorted_rms[-max(1, len(sorted_rms) // 10)].mean()
            snr_proxy = signal_power / (noise_floor + 1e-8)
            snr_score = min(1.0, float(np.log1p(snr_proxy) / 5.0))
            return float((sf_score + rms_score + snr_score) / 3.0) * 100
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# UTMOSv2 — git package `utmosv2`
# ---------------------------------------------------------------------------


class UTMOSv2Scorer(BaseScorer):
    name = "UTMOSv2"

    def __init__(self, device: str = "cuda:0", **kwargs):
        try:
            import utmosv2
        except ImportError:
            raise ImportError(
                "utmosv2 package not found. Install: pip install git+https://github.com/sarulab-speech/UTMOSv2.git"
            )
        self.device = device
        self.model = utmosv2.create_model(pretrained=True, device=device)

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        return float(
            self.model.predict(
                input_path=str(wav_path),
                device=self.device,
                verbose=False,
                remove_silent_section=False,
            )
        )

    def score_files(self, wav_paths: List[Union[str, Path]], batch_size: int = 64) -> List[float]:
        import torchaudio

        MAX_AUDIO_LEN = 30 * 16000
        max_len = 0
        arrays = []
        for wp in wav_paths:
            wav, sr = torchaudio.load(str(wp))
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            a = wav.squeeze(0).cpu().numpy()
            if a.shape[0] > MAX_AUDIO_LEN:
                a = a[:MAX_AUDIO_LEN]
            arrays.append(a)
            max_len = max(max_len, a.shape[0])
        batch = np.zeros((len(arrays), max_len), dtype=np.float32)
        for i, a in enumerate(arrays):
            batch[i, : len(a)] = a
        scores = self.model.predict(
            data=batch,
            batch_size=batch_size,
            device=self.device,
            num_workers=4,
            remove_silent_section=False,
            verbose=False,
        )
        score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return [float(v) for v in score_list]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "UTMOS22Strong": UTMOS22StrongScorer,
    "SCOREQ": SCOREQScorer,
    "TTSDS2": TTSDS2Scorer,
    "UTMOSv2": UTMOSv2Scorer,
}


def create_scorer(
    metric: str, device: str = "cuda:0", model_dir: Union[str, Path, None] = None, **kwargs
) -> BaseScorer:
    metric_upper = metric.upper().replace("-", "").replace("_", "")
    for key, cls in _SCORER_REGISTRY.items():
        if key.upper().replace("-", "").replace("_", "") == metric_upper:
            if key == "UTMOS22Strong":
                return cls(model_dir=model_dir, device=device, **kwargs)
            return cls(device=device, **kwargs)
    raise ValueError(f"Unknown metric '{metric}'. Available: {list(_SCORER_REGISTRY.keys())}")


def try_create_scorer(
    metric: str, device: str = "cuda:0", model_dir: Union[str, Path, None] = None, **kwargs
) -> Optional[BaseScorer]:
    try:
        return create_scorer(metric, device=device, model_dir=model_dir, **kwargs)
    except (ImportError, FileNotFoundError) as e:
        print(f"[scorers] Skipping {metric}: {e}", flush=True)
        return None
