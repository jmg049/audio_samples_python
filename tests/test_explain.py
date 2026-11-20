"""Tests for the explain API (``audio_samples.explain``, ``ExplainDoc``).

Every number an explanation reports is checked against an independent NumPy /
SciPy calculation or against the result of actually running the operation
(frame counts from ``audio.stft``, VAD frames from ``voice_activity_mask``,
empty mel bands from ``mel_spectrogram``).
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import explain as ex
from audio_samples.explain import ExplainDoc

W = aus.WindowType


def _all_docs() -> dict[str, ExplainDoc]:
    """One doc from every explainer, for the rendering invariants."""
    audio = aus.sine_wave(440.0, 1.0, sample_rate=16000)
    stft = aus.StftParams(n_fft=512, hop_size=160, window=W.hanning)
    return {
        "stft": audio.explain(stft, f0_range=(80.0, 300.0)),
        "spectrogram": ex.explain(aus.SpectrogramParams(stft, 16000.0), n_samples=16000),
        "mel": ex.explain(aus.MelParams(80, 0.0, 8000.0), sample_rate=16000, n_fft=512),
        "mfcc": ex.explain(aus.MfccParams(13), sample_rate=16000, n_mels=40),
        "window": ex.explain(W.blackman, sample_rate=16000),
        "iir": aus.IirFilterDesign.butterworth_lowpass(4, 1000.0).explain(48000.0),
        "eq": aus.EqBand.peak(1000.0, 6.0, 1.0).explain(48000.0),
        "compressor": aus.CompressorConfig.default().explain(48000.0),
        "limiter": aus.LimiterConfig.default().explain(48000.0),
        "vad": aus.VadConfig.default().explain(16000.0, n_samples=16000),
        "onset": aus.OnsetDetectionConfig.percussive(44100).explain(44100.0),
        "sampling": ex.sampling(44100.0),
        "aliasing": ex.aliasing(7000.0, 8000.0),
        "quantisation": ex.quantisation(16),
        "dft": ex.dft(1024, 16000.0),
        "topic_window": ex.window(W.hanning, 1024, 16000.0),
        "yin": ex.yin(0.15, 50.0, 500.0, 16000.0, 1024),
        "autocorrelation": ex.autocorrelation_pitch(50.0, 500.0, 16000.0),
        "lpc": ex.lpc(12, 16000.0),
        "cepstrum": ex.cepstrum(1024, 16000.0),
        "resampling": ex.resampling(48000.0, 16000.0),
        "fir": ex.fir([0.25, 0.5, 0.25], 16000.0),
    }


ALL_DOCS = _all_docs()


# ---------------------------------------------------------------------------
# Module layout and the ExplainDoc object
# ---------------------------------------------------------------------------


def test_module_layout():
    assert sys.modules["audio_samples.explain"] is aus.explain
    assert aus.ExplainDoc is ex.ExplainDoc
    assert "explain" in aus.__all__ and "ExplainDoc" in aus.__all__
    for name in ex.__all__:
        assert hasattr(ex, name), name
    from audio_samples.explain import explain, fir  # noqa: F401  (importable)


def test_import_does_not_load_a_browser():
    code = "import sys, audio_samples; print('webbrowser' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("name", sorted(ALL_DOCS))
def test_doc_structure(name):
    doc = ALL_DOCS[name]
    assert isinstance(doc, ExplainDoc)
    assert doc.title and doc.summary and isinstance(doc.topic, str)
    for f in doc.formulas:
        assert set(f) == {"latex", "caption"}
    for s in doc.symbols:
        assert set(s) == {"symbol", "meaning", "value", "unit"}
    for q in doc.derived:
        assert set(q) == {"name", "symbol", "value", "unit", "how"}
        assert isinstance(q["value"], float) and math.isfinite(q["value"])
        assert doc.quantity(q["name"]) is not None
    for t in doc.tables:
        assert set(t) == {"caption", "headers", "rows"}
        assert all(len(r) == len(t["headers"]) for r in t["rows"])
    for w in doc.warnings:
        assert set(w) == {"level", "message", "hint"}
        assert w["level"] in {"info", "caution", "problem"}
        assert doc.has_warning(w["level"])
    for s in doc.see_also:
        assert set(s) == {"label", "target"}
    assert all(isinstance(t, str) for t in doc.trade_offs)


@pytest.mark.parametrize("name", sorted(ALL_DOCS))
def test_renderings(name):
    doc = ALL_DOCS[name]
    text = doc.text()
    assert str(doc) == text
    assert "\\frac" not in text and "\\" not in text, "raw LaTeX in text output"
    md = doc.markdown()
    assert md.startswith("### ")
    if doc.formulas:
        assert "$$" in md
    assert doc._repr_markdown_() == md
    html = doc.html()
    assert "<script" not in html.lower()
    r = repr(doc)
    assert "\n" not in r and doc.title in r


def test_no_repr_html_so_jupyter_uses_markdown():
    # Jupyter prefers text/html over text/markdown; HTML would lose the $$ maths.
    assert not hasattr(ExplainDoc, "_repr_html_")


def test_quantity_and_has_warning_lookups():
    doc = ex.aliasing(7000.0, 8000.0)
    assert doc.quantity("no such quantity") is None
    assert doc.has_warning("caution") and doc.has_warning("CAUTION")
    assert not doc.has_warning("problem")
    assert doc.has_warning()
    assert doc.has_warning_containing("alias")
    with pytest.raises(ValueError, match="unknown warning level"):
        doc.has_warning("severe")


def test_python_see_also_targets_exist():
    for doc in ALL_DOCS.values():
        for s in doc.see_also:
            t = s["target"]
            if not t.startswith("audio_samples.") or "::" in t:
                continue
            obj = aus
            for part in t.split("(")[0].split(".")[1:]:
                obj = getattr(obj, part)


def test_unsupported_type_raises_type_error():
    with pytest.raises(TypeError, match="supported types: StftParams"):
        ex.explain(42)
    with pytest.raises(TypeError):
        aus.sine_wave(440.0, 0.1, sample_rate=8000).explain("stft")
    with pytest.raises(TypeError):
        ex.window("hann", 256, 16000.0)


# ---------------------------------------------------------------------------
# Numbers against independent calculations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_fft,hop,fs", [(512, 160, 16000), (1024, 256, 44100), (256, 128, 8000)])
def test_stft_numbers_match_numpy_and_real_stft(n_fft, hop, fs):
    audio = aus.white_noise(0.73, sample_rate=fs, amplitude=0.1)
    params = aus.StftParams(n_fft=n_fft, hop_size=hop, window=W.hanning)
    doc = audio.explain(params)
    assert doc.quantity("Frequency resolution") == pytest.approx(fs / n_fft)
    assert doc.quantity("Frequency bins") == n_fft // 2 + 1
    assert doc.quantity("Frame period") == pytest.approx(1000.0 * hop / fs)
    assert doc.quantity("Window duration") == pytest.approx(1000.0 * n_fft / fs)
    assert doc.quantity("Overlap") == pytest.approx(100.0 * (1 - hop / n_fft))
    result = audio.stft(params)
    assert doc.quantity("Frames") == result.n_frames
    assert doc.quantity("Frequency bins") == result.n_bins
    # Same numbers through the dispatcher with explicit context.
    doc2 = ex.explain(params, sample_rate=fs, n_samples=audio.samples_per_channel())
    assert doc2.quantity("Frames") == result.n_frames


def test_stft_cola_figures_match_numpy():
    n, hop = 256, 128
    w = np.hanning(n)  # symmetric Hann, like the crate's window
    total = np.zeros(hop * 64 + n)
    for m in range(64):
        total[m * hop : m * hop + n] += w
    steady = total[n : n + hop]  # one hop in the fully overlapped region
    doc = ex.explain(aus.StftParams(n_fft=n, hop_size=hop, window=W.hanning), sample_rate=16000)
    assert doc.quantity("Overlap-add gain") == pytest.approx(steady.mean(), rel=1e-6)
    ripple = (steady.max() - steady.min()) / steady.mean()
    assert doc.quantity("Overlap-add ripple") == pytest.approx(ripple, rel=1e-4)


def test_stft_short_window_warning_for_low_f0():
    params = aus.StftParams(n_fft=256, hop_size=128, window=W.hanning)
    doc = ex.explain(params, sample_rate=16000, f0_range=(80.0, 300.0))
    assert doc.has_warning("caution")
    assert doc.has_warning_containing("shorter than two periods")
    # 2 periods of 80 Hz = 25 ms = 400 samples: a 512-point window is long enough.
    ok = ex.explain(
        aus.StftParams(n_fft=512, hop_size=128, window=W.hanning),
        sample_rate=16000,
        f0_range=(80.0, 300.0),
    )
    assert not ok.has_warning_containing("shorter than two periods")


def test_hann_window_figures_match_numpy():
    n, fs = 1024, 16000.0
    w = np.hanning(n)
    enbw = n * np.sum(w**2) / np.sum(w) ** 2
    doc = ex.window(W.hanning, n, fs)
    assert doc.quantity("Equivalent noise bandwidth") == pytest.approx(enbw, rel=1e-9)
    assert doc.quantity("Equivalent noise bandwidth") == pytest.approx(1.5, abs=0.002)
    assert doc.quantity("Coherent gain") == pytest.approx(w.mean(), rel=1e-9)
    assert doc.quantity("ENBW (Hz)") == pytest.approx(enbw * fs / n, rel=1e-9)
    # Peak side lobe from a heavily zero-padded FFT.
    pad = 64
    spec = np.abs(np.fft.rfft(w, n * pad))
    spec_db = 20 * np.log10(spec / spec[0] + 1e-300)
    first_null = int(np.argmax(np.diff(spec_db) > 0))
    psl = spec_db[first_null : first_null + 4 * pad].max()
    assert doc.quantity("Peak side-lobe level") == pytest.approx(psl, abs=0.02)
    assert doc.quantity("First null") == pytest.approx(first_null / pad, abs=0.05)
    # Without a sample rate (dispatcher on the bare window) the figures in bins agree.
    assert ex.explain(W.hanning).quantity("Equivalent noise bandwidth") == pytest.approx(
        enbw, rel=1e-9
    )


@pytest.mark.parametrize(
    "f,fs", [(7000.0, 8000.0), (1000.0, 8000.0), (11000.0, 8000.0), (44100.0 * 0.6, 44100.0)]
)
def test_aliasing_matches_folding_formula(f, fs):
    expected = abs(f - fs * round(f / fs))
    doc = ex.aliasing(f, fs)
    assert doc.quantity("Alias frequency") == pytest.approx(expected)
    assert doc.quantity("Nyquist frequency") == fs / 2
    assert doc.has_warning("caution") == (f > fs / 2)


def test_aliasing_7k_at_8k():
    assert ex.aliasing(7000.0, 8000.0).quantity("Alias frequency") == 1000.0


@pytest.mark.parametrize("bits", [8, 16, 24])
def test_quantisation_sqnr(bits):
    expected = 10 * np.log10(1.5 * 2.0 ** (2 * bits))
    doc = ex.quantisation(bits)
    assert doc.quantity("SQNR (full-scale sine)") == pytest.approx(expected, rel=1e-12)
    assert doc.quantity("Quantisation levels") == 2**bits


def test_quantisation_16_bit_is_98_09_db():
    assert ex.quantisation(16).quantity("SQNR (full-scale sine)") == pytest.approx(98.09, abs=0.005)


def test_sampling_dft_cepstrum_resampling_numbers():
    assert ex.sampling(16000.0).quantity("Nyquist frequency") == 8000.0
    assert ex.sampling(16000.0).quantity("Sampling period") == pytest.approx(1000.0 / 16000)
    d = ex.dft(512, 16000.0)
    assert d.quantity("Bin spacing") == pytest.approx(16000 / 512)
    assert d.quantity("Non-redundant bins") == len(np.fft.rfftfreq(512))
    assert d.quantity("Highest bin frequency") == pytest.approx(np.fft.rfftfreq(512, 1 / 16000)[-1])
    c = ex.cepstrum(1024, 16000.0)
    assert c.quantity("Quefrency resolution") == pytest.approx(1000.0 / 16000)
    # The real cepstrum is symmetric: only N/2 quefrencies are distinct.
    assert c.quantity("Longest quefrency") == pytest.approx(1000.0 * (1024 / 2) / 16000)
    assert c.quantity("Lowest detectable f0") == pytest.approx(2 * 16000 / 1024)
    r = ex.resampling(48000.0, 16000.0)
    assert r.quantity("Ratio") == pytest.approx(1 / 3)
    assert r.quantity("Output Nyquist") == 8000.0
    assert r.has_warning("caution")  # downsampling: anti-alias filtering matters


def test_pitch_topics_lag_range():
    fs, fmin, fmax = 16000.0, 50.0, 500.0
    y = ex.yin(0.15, fmin, fmax, fs, 1024)
    assert y.quantity("Longest lag") == math.floor(fs / fmin)
    assert y.quantity("Shortest lag") == math.floor(fs / fmax)
    assert not y.has_warning("problem")
    too_short = ex.yin(0.15, 20.0, 500.0, fs, 1024)  # 20 Hz needs 1602 samples
    assert too_short.has_warning("problem")
    ac = ex.autocorrelation_pitch(fmin, fmax, fs)
    assert ac.quantity("Longest lag") == math.floor(fs / fmin)


def test_fir_numbers_match_numpy():
    rng = np.random.default_rng(1)
    half = rng.normal(size=8)
    h = np.concatenate([half, half[::-1]])  # symmetric, even length: type II
    fs = 16000.0
    doc = ex.fir(h, fs)
    assert doc.quantity("Taps") == len(h)
    assert doc.quantity("DC gain") == pytest.approx(abs(h.sum()))
    nyq = abs(np.sum(h * (-1.0) ** np.arange(len(h))))
    assert doc.quantity("Nyquist gain") == pytest.approx(nyq, abs=1e-12)
    assert doc.quantity("Group delay") == pytest.approx((len(h) - 1) / 2)
    assert doc.has_warning_containing("Linear-phase")
    # A list works as well as an array.
    assert ex.fir(list(h), fs).quantity("Taps") == len(h)


def test_iir_butterworth_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    fs, fc = 48000.0, 1000.0
    design = aus.IirFilterDesign.butterworth_lowpass(4, fc)
    doc = design.explain(fs)
    sos = design.sos(fs)
    _, h = signal.sosfreqz(sos, worN=[fc], fs=fs)
    gain_db = 20 * np.log10(np.abs(h[0]))
    assert gain_db == pytest.approx(-3.0103, abs=1e-3)
    assert doc.quantity("Measured gain at the cutoff") == pytest.approx(gain_db, abs=1e-9)
    _, poles, _ = signal.sos2zpk(sos)
    assert doc.quantity("Largest pole radius") == pytest.approx(np.max(np.abs(poles)), rel=1e-9)
    assert doc.quantity("Sections") == sos.shape[0]
    assert not doc.has_warning("problem")
    # Same result through the dispatcher and through a signal.
    audio = aus.sine_wave(440.0, 0.1, sample_rate=48000)
    assert audio.explain(design).quantity("Measured gain at the cutoff") == pytest.approx(gain_db)
    assert ex.explain(design, sample_rate=fs).quantity("Largest pole radius") == pytest.approx(
        doc.quantity("Largest pole radius")
    )


def test_iir_cutoff_above_nyquist_is_a_problem():
    doc = aus.IirFilterDesign.butterworth_lowpass(4, 30000.0).explain(48000.0)
    assert doc.has_warning("problem")


def test_explain_without_sample_rate_notes_it():
    doc = aus.IirFilterDesign.butterworth_lowpass(4, 1000.0).explain()
    assert doc.has_warning("info")
    assert doc.quantity("Measured gain at the cutoff") is None


def test_eq_band_peak_gain():
    signal = pytest.importorskip("scipy.signal")
    fs, f0, gain = 48000.0, 1000.0, 6.0
    doc = aus.EqBand.peak(f0, gain, 1.0).explain(fs)
    assert doc.quantity("Measured gain at f0") == pytest.approx(gain, abs=1e-6)
    # RBJ cookbook peaking EQ, computed independently.
    a = 10 ** (gain / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * 1.0)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    assert doc.quantity("Amplitude parameter") == pytest.approx(a)
    poles = np.roots(den)
    assert doc.quantity("Largest pole radius") == pytest.approx(np.max(np.abs(poles)), rel=1e-6)
    _, h = signal.freqz(b, den, worN=[0.0], fs=fs)
    assert doc.quantity("Measured gain at DC") == pytest.approx(20 * np.log10(abs(h[0])), abs=1e-6)


def test_compressor_numbers():
    cfg = aus.CompressorConfig.default()
    # CompressorConfig has no field getters; read the defaults from its repr.
    r = repr(cfg)
    for field in ("threshold_db: -12.0", "ratio: 4.0", "attack_ms: 5.0", "release_ms: 50.0"):
        assert field in r, r
    threshold_db, ratio, attack_ms, release_ms = -12.0, 4.0, 5.0, 50.0
    fs = 48000.0
    doc = cfg.explain(fs)
    assert doc.quantity("Attack coefficient") == pytest.approx(
        math.exp(-1.0 / (attack_ms / 1000.0 * fs))
    )
    assert doc.quantity("Release coefficient") == pytest.approx(
        math.exp(-1.0 / (release_ms / 1000.0 * fs))
    )
    assert doc.quantity("Attack 10-90% rise time") == pytest.approx(attack_ms * math.log(9))
    # 0 dBFS is well above threshold + knee/2: the hard-knee formula applies.
    expected_gr = (0.0 - threshold_db) * (1 - 1 / ratio)
    assert doc.quantity("Gain reduction at 0 dBFS") == pytest.approx(expected_gr)
    assert doc.quantity("Slope above the knee") == pytest.approx(1 / ratio)
    assert doc.tables and doc.tables[0]["headers"][0].startswith("Input level")


def test_vad_frame_count_matches_mask():
    cfg = aus.VadConfig.default()
    audio = aus.white_noise(1.0, sample_rate=16000, amplitude=0.1)
    doc = audio.explain(cfg)
    assert doc.quantity("Frames") == len(audio.voice_activity_mask(cfg))


def test_mel_empty_bands_match_real_mel_spectrogram():
    fs, n_fft, n_mels = 16000, 256, 128
    stft = aus.StftParams(n_fft=n_fft, hop_size=128, window=W.hanning)
    mel = aus.MelParams(n_mels, 0.0, 8000.0)
    doc = ex.explain(mel, sample_rate=fs, n_fft=n_fft)
    audio = aus.white_noise(0.5, sample_rate=fs, amplitude=0.5)
    spec = np.asarray(audio.mel_spectrogram(aus.SpectrogramParams(stft, float(fs)), mel).data)
    empty_rows = int(np.sum(np.all(spec == 0.0, axis=1)))
    assert empty_rows > 0
    assert doc.quantity("Empty filters") == empty_rows
    assert doc.has_warning("caution") or doc.has_warning("problem")
    assert doc.quantity("FFT bin spacing") == pytest.approx(fs / n_fft)


def test_mfcc_more_coefficients_than_mels_warns():
    doc = ex.explain(aus.MfccParams(40), n_mels=20)
    assert doc.has_warning("problem") or doc.has_warning("caution")
    fine = ex.explain(aus.MfccParams(13), n_mels=40)
    assert fine.quantity("Output coefficients per frame") == 13


def test_audio_explain_uses_signal_context():
    stereo = aus.stereo_sine_wave(440.0, 0.5, sample_rate=22050)
    params = aus.StftParams(n_fft=512, hop_size=256, window=W.hanning)
    doc = stereo.explain(params)
    assert doc.quantity("Frequency resolution") == pytest.approx(22050 / 512)
    # stft() is mono-only; a mono signal of the same length gives the frame count.
    mono = aus.sine_wave(440.0, 0.5, sample_rate=22050)
    assert mono.samples_per_channel() == stereo.samples_per_channel()
    assert doc.quantity("Frames") == mono.stft(params).n_frames
    with_f0 = stereo.explain(params, f0_range=(60.0, 200.0))
    assert with_f0.has_warning_containing("shorter than two periods")


def test_config_explain_methods_exist():
    for cls in (
        aus.IirFilterDesign,
        aus.EqBand,
        aus.CompressorConfig,
        aus.LimiterConfig,
        aus.VadConfig,
        aus.OnsetDetectionConfig,
    ):
        assert callable(getattr(cls, "explain")), cls
