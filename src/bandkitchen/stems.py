"""Source separation for play-along splicing: take the player's instrument out of the song.

Two separators share one interface (``separate(audio) -> {stem_name: audio}``):

- ``DemucsSeparator`` (default, recommended): Meta's Hybrid Transformer Demucs, a neural
  network trained to split music into drums / bass / vocals / other (the 6-stem model also
  gives guitar and piano). Drums and bass separate very cleanly; guitar and piano are
  noticeably rougher. Needs the optional ``stems`` extra (PyTorch); the first run downloads
  the model weights (~80 MB). Roughly real-time-ish on an Apple-silicon CPU, so results are
  cached per song (see ``cache.py``).
- ``HpssSeparator`` (fallback, drums only): harmonic/percussive separation by median
  filtering (Fitzgerald, 2010), pure numpy/scipy. Sustained tones form horizontal ridges
  in a spectrogram and hits form vertical ones, so a median along time keeps the sustained
  part and a median along frequency keeps the hits. It removes cymbals and snare well but
  leaves some kick body and softens other instruments' attacks (guitar strums, piano
  hammers). Useful with no ML install; clearly worse than Demucs.

All audio here is float32 stereo, shape (n, 2), at the separator's ``sample_rate``.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
import scipy.signal as ss
from scipy.ndimage import median_filter

INSTRUMENTS = ("drums", "bass", "vocals", "guitar", "piano", "other")


class SeparatorUnavailable(RuntimeError):
    pass


class Separator(Protocol):
    sample_rate: int

    @property
    def cache_id(self) -> str:
        """Identifies the separator + settings, so cached stems are never mixed up."""
        ...

    def stems(self) -> tuple[str, ...]: ...

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]: ...


def remove_instrument(stems: dict[str, np.ndarray], instrument: str) -> np.ndarray:
    """The song without ``instrument``: the sum of every other stem.

    Summing the remaining stems (rather than subtracting the removed one from the original)
    leaves no trace of the removed instrument's separation errors in the bed.
    """
    if instrument not in stems:
        raise ValueError(
            f"this separator can't isolate {instrument!r}; it provides {sorted(stems)}"
        )
    others = [v for k, v in stems.items() if k != instrument]
    if not others:
        raise ValueError("nothing left after removing the only stem")
    return np.sum(others, axis=0).astype(np.float32)


class HpssSeparator:
    """Median-filter harmonic/percussive separation. Stems: 'drums' (percussive), 'other'."""

    sample_rate = 44100

    def __init__(self, n_fft: int = 4096, kernel: int = 17, power: float = 2.0):
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.kernel = kernel  # median length: frames (time) and bins (frequency)
        self.power = power  # soft-mask sharpness; higher = more decisive split

    @property
    def cache_id(self) -> str:
        return f"hpss-n{self.n_fft}-k{self.kernel}-p{self.power:g}"

    def stems(self) -> tuple[str, ...]:
        return ("drums", "other")

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        kw = dict(fs=self.sample_rate, nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
        _, _, z = ss.stft(x.T, **kw)  # (2, freq, time)
        # One mask from the mid (L+R) magnitude keeps the stereo image intact.
        mag = np.abs(z).mean(axis=0).astype(np.float32)
        harm = median_filter(mag, size=(1, self.kernel), mode="nearest")  # along time
        perc = median_filter(mag, size=(self.kernel, 1), mode="nearest")  # along frequency
        hp, pp = harm**self.power, perc**self.power
        mask_p = pp / (hp + pp + 1e-12)
        out = {}
        for name, m in (("drums", mask_p), ("other", 1.0 - mask_p)):
            _, y = ss.istft(z * m[None], **kw)
            out[name] = y.T[: len(x)].astype(np.float32)
        return out


class DemucsSeparator:
    """Hybrid Transformer Demucs. ``model``: 'htdemucs' (4 stems) or 'htdemucs_6s' (6)."""

    sample_rate = 44100

    def __init__(self, model: str = "htdemucs", device: str | None = None, shifts: int = 1):
        self.model_name = model
        self.device = device
        self.shifts = shifts  # >1 averages random time shifts: slightly better, N times slower
        self._model = None

    @property
    def cache_id(self) -> str:
        return f"demucs-{self.model_name}-s{self.shifts}"

    def stems(self) -> tuple[str, ...]:
        if self.model_name == "htdemucs_6s":
            return ("drums", "bass", "other", "vocals", "guitar", "piano")
        return ("drums", "bass", "other", "vocals")

    def _load(self):
        if self._model is None:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from demucs.pretrained import get_model
            except ImportError as e:
                raise SeparatorUnavailable(
                    'Demucs is not installed. Run: pip install -e ".[stems]" '
                    "(or use --separator hpss for a rough, drums-only fallback)"
                ) from e
            self._torch = torch
            self._model = get_model(self.model_name).eval()
            self.sample_rate = int(self._model.samplerate)
        return self._model

    def _devices(self) -> list[str]:
        if self.device:
            return [self.device]
        torch = self._torch
        if torch.cuda.is_available():
            return ["cuda", "cpu"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return ["mps", "cpu"]
        return ["cpu"]

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        model = self._load()
        from demucs.apply import apply_model

        torch = self._torch
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(x.T))
        # Same normalization as the demucs CLI: zero-mean, unit-std on the mono reference.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        last_error = None
        for device in self._devices():
            try:
                with torch.no_grad():
                    out = apply_model(
                        model,
                        wav[None],
                        device=device,
                        shifts=self.shifts,
                        split=True,
                        overlap=0.25,
                        progress=False,
                    )[0]
                break
            except RuntimeError as e:  # e.g. an op missing on Apple GPU: fall back to CPU
                last_error = e
        else:
            raise SeparatorUnavailable(f"Demucs failed on all devices: {last_error}")
        out = out * std + mean
        return {
            name: out[i].cpu().numpy().T.astype(np.float32) for i, name in enumerate(model.sources)
        }


def make_separator(kind: str, instrument: str | None = None) -> Separator:
    """``kind``: 'demucs' or 'hpss'. Picks the 6-stem Demucs model for guitar/piano."""
    if kind == "hpss":
        return HpssSeparator()
    if kind == "demucs":
        model = "htdemucs_6s" if instrument in ("guitar", "piano") else "htdemucs"
        return DemucsSeparator(model=model)
    raise ValueError(f"unknown separator {kind!r}; use 'demucs' or 'hpss'")
