"""
Filtering, with scipy.signal as the reference implementation:

1. IIR designs (Butterworth, Chebyshev I/II, elliptic, Bessel) produce the
   same second-order sections as scipy.signal
2. Applying a design matches scipy.signal.sosfilt / sosfiltfilt
3. SosFilter interoperates with scipy SOS arrays and streams correctly
4. The simple first-order filters match scipy.signal.butter(1, ...)
5. Parametric EQ bands match the RBJ Audio EQ Cookbook
6. Integer (int16) audio is filtered without bias or silent failure
7. Copy / read-only / layout semantics are safe

apply_* / butterworth_* / filtfilt modify the AudioSamples in place;
low_pass_filter / high_pass_filter / band_pass_filter return a new object.
AudioSamples wraps NumPy arrays without copying; use .copy() to keep one.
"""

import warnings

import numpy as np
import pytest
import scipy.signal as sg

import audio_samples as aus
FilterResponse = aus.FilterResponse
IirFilterDesign = aus.IirFilterDesign
SosFilter = aus.SosFilter

# Tolerance constants
TIGHT_RTOL = 1e-6
TIGHT_ATOL = 1e-7
# Coefficients and float64 outputs agree with scipy to near machine precision.
SCIPY_ATOL = 1e-10

FS = 48000


@pytest.fixture
def noise():
    """Deterministic broadband test signal (float64, mono)."""
    rng = np.random.default_rng(1234)
    return rng.uniform(-0.5, 0.5, 4096) + 0.05


def assert_sos_equal(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{actual.shape} vs {expected.shape}"
    scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1.0)
    err = np.max(np.abs(actual - expected) / scale)
    assert err < 1e-10, f"max relative coefficient error {err:.3e}"


# (name, audio_samples design factory, scipy sos factory)
DESIGNS = [
    ("butter_lp", lambda n: IirFilterDesign.butterworth_lowpass(n, 1000.0),
     lambda n: sg.butter(n, 1000.0, "lowpass", fs=FS, output="sos")),
    ("butter_hp", lambda n: IirFilterDesign.butterworth_highpass(n, 100.0),
     lambda n: sg.butter(n, 100.0, "highpass", fs=FS, output="sos")),
    ("butter_bp", lambda n: IirFilterDesign.butterworth_bandpass(n, 300.0, 3400.0),
     lambda n: sg.butter(n, [300.0, 3400.0], "bandpass", fs=FS, output="sos")),
    ("butter_bs", lambda n: IirFilterDesign.butterworth_bandstop(n, 50.0, 60.0),
     lambda n: sg.butter(n, [50.0, 60.0], "bandstop", fs=FS, output="sos")),
    ("cheby1_lp", lambda n: IirFilterDesign.chebyshev_i("lowpass", n, 5000.0, 1.0),
     lambda n: sg.cheby1(n, 1.0, 5000.0, "lowpass", fs=FS, output="sos")),
    ("cheby1_hp", lambda n: IirFilterDesign.chebyshev_i(FilterResponse.highpass, n, 200.0, 0.5),
     lambda n: sg.cheby1(n, 0.5, 200.0, "highpass", fs=FS, output="sos")),
    ("cheby1_bp", lambda n: IirFilterDesign.chebyshev_i_band("bandpass", n, 500.0, 2000.0, 0.1),
     lambda n: sg.cheby1(n, 0.1, [500.0, 2000.0], "bandpass", fs=FS, output="sos")),
    ("cheby2_lp", lambda n: IirFilterDesign.chebyshev_ii("lowpass", n, 8000.0, 60.0),
     lambda n: sg.cheby2(n, 60.0, 8000.0, "lowpass", fs=FS, output="sos")),
    ("cheby2_bs", lambda n: IirFilterDesign.chebyshev_ii_band("bandstop", n, 900.0, 1100.0, 40.0),
     lambda n: sg.cheby2(n, 40.0, [900.0, 1100.0], "bandstop", fs=FS, output="sos")),
    ("ellip_lp", lambda n: IirFilterDesign.elliptic("lowpass", n, 4000.0, 0.5, 60.0),
     lambda n: sg.ellip(n, 0.5, 60.0, 4000.0, "lowpass", fs=FS, output="sos")),
    ("ellip_hp", lambda n: IirFilterDesign.elliptic("highpass", n, 4000.0, 1.0, 40.0),
     lambda n: sg.ellip(n, 1.0, 40.0, 4000.0, "highpass", fs=FS, output="sos")),
    ("ellip_bp", lambda n: IirFilterDesign.elliptic_band("bandpass", n, 1000.0, 3000.0, 1.0, 50.0),
     lambda n: sg.ellip(n, 1.0, 50.0, [1000.0, 3000.0], "bandpass", fs=FS, output="sos")),
    ("bessel_lp_mag", lambda n: IirFilterDesign.bessel("lowpass", n, 2000.0),
     lambda n: sg.bessel(n, 2000.0, "lowpass", norm="mag", fs=FS, output="sos")),
    ("bessel_lp_phase", lambda n: IirFilterDesign.bessel("lowpass", n, 2000.0, norm="phase"),
     lambda n: sg.bessel(n, 2000.0, "lowpass", fs=FS, output="sos")),
    ("bessel_bp_delay", lambda n: IirFilterDesign.bessel_band("bandpass", n, 500.0, 5000.0, norm="delay"),
     lambda n: sg.bessel(n, [500.0, 5000.0], "bandpass", norm="delay", fs=FS, output="sos")),
]


class TestIirDesignMatchesScipy:
    """IirFilterDesign.sos() must equal the scipy.signal design."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 8])
    @pytest.mark.parametrize("name,ours,theirs", DESIGNS, ids=[d[0] for d in DESIGNS])
    def test_sos_coefficients(self, name, ours, theirs, order):
        assert_sos_equal(ours(order).sos(FS), theirs(order))

    @pytest.mark.parametrize("name,ours,theirs", DESIGNS, ids=[d[0] for d in DESIGNS])
    def test_frequency_response(self, name, ours, theirs):
        freqs = np.geomspace(10.0, FS / 2 * 0.999, 64)
        mag, phase = ours(4).frequency_response(freqs, FS)
        _, h = sg.sosfreqz(theirs(4), worN=freqs, fs=FS)
        np.testing.assert_allclose(mag, np.abs(h), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(np.exp(1j * phase), h / np.maximum(np.abs(h), 1e-300), atol=1e-6)

    def test_generic_constructor_accepts_ripple_and_attenuation(self):
        design = IirFilterDesign(
            aus.types.IirFilterType.elliptic, "bandstop", 3,
            low_frequency=1000.0, high_frequency=2000.0,
            passband_ripple=1.0, stopband_attenuation=50.0,
        )
        assert design.passband_ripple == 1.0
        assert design.stopband_attenuation == 50.0
        assert_sos_equal(design.sos(FS), sg.ellip(3, 1.0, 50.0, [1000.0, 2000.0], "bandstop", fs=FS, output="sos"))

    def test_invalid_designs_raise_value_error(self):
        with pytest.raises(ValueError):
            IirFilterDesign.butterworth_lowpass(4, FS).sos(FS)  # cutoff at fs (> Nyquist)
        with pytest.raises(ValueError):
            IirFilterDesign.chebyshev_i("lowpass", 4, 1000.0, -1.0).sos(FS)
        with pytest.raises(ValueError):
            IirFilterDesign.chebyshev_i("sideways", 4, 1000.0, 1.0)


class TestFilteringMatchesScipy:
    """Filtering AudioSamples must equal scipy.signal.sosfilt / sosfiltfilt."""

    @pytest.mark.parametrize("name,ours,theirs", DESIGNS, ids=[d[0] for d in DESIGNS])
    def test_apply_iir_filter(self, name, ours, theirs, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        audio.apply_iir_filter(ours(4))
        np.testing.assert_allclose(audio.to_numpy(), sg.sosfilt(theirs(4), noise), atol=SCIPY_ATOL)

    @pytest.mark.parametrize("name,ours,theirs", DESIGNS, ids=[d[0] for d in DESIGNS])
    def test_filtfilt(self, name, ours, theirs, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        audio.filtfilt(ours(3))
        np.testing.assert_allclose(audio.to_numpy(), sg.sosfiltfilt(theirs(3), noise), atol=1e-9)

    def test_multichannel_channels_are_independent(self, noise):
        stereo = np.stack([noise, -0.5 * noise, noise[::-1].copy(), np.zeros_like(noise)])
        audio = aus.AudioSamples.new_multi(stereo.copy(), FS)
        audio.apply_iir_filter(IirFilterDesign.butterworth_highpass(4, 100.0))
        expected = sg.sosfilt(sg.butter(4, 100.0, "highpass", fs=FS, output="sos"), stereo, axis=-1)
        np.testing.assert_allclose(audio.to_numpy(), expected, atol=SCIPY_ATOL)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 8])
    def test_butterworth_convenience_methods(self, order, noise):
        for method, args, btype, wn in [
            ("apply_butterworth_lowpass", (1000.0,), "lowpass", 1000.0),
            ("apply_butterworth_highpass", (80.0,), "highpass", 80.0),
            ("apply_butterworth_bandpass", (300.0, 3400.0), "bandpass", [300.0, 3400.0]),
            ("butterworth_highpass", (80.0,), "highpass", 80.0),
        ]:
            audio = aus.AudioSamples.new_mono(noise.copy(), FS)
            getattr(audio, method)(order, *args)
            expected = sg.sosfilt(sg.butter(order, wn, btype, fs=FS, output="sos"), noise)
            np.testing.assert_allclose(audio.to_numpy(), expected, atol=SCIPY_ATOL, err_msg=method)

    @pytest.mark.parametrize("response", ["highpass", FilterResponse.highpass])
    def test_apply_chebyshev_i_accepts_str_or_enum(self, response, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        audio.apply_chebyshev_i(5, 500.0, 1.0, response)
        expected = sg.sosfilt(sg.cheby1(5, 1.0, 500.0, "highpass", fs=FS, output="sos"), noise)
        np.testing.assert_allclose(audio.to_numpy(), expected, atol=SCIPY_ATOL)

    def test_family_apply_methods(self, noise):
        cases = [
            (lambda a: a.apply_chebyshev_ii_lowpass(6, 5000.0, 60.0),
             sg.cheby2(6, 60.0, 5000.0, "lowpass", fs=FS, output="sos")),
            (lambda a: a.apply_elliptic_bandpass(3, 1000.0, 3000.0, 1.0, 50.0),
             sg.ellip(3, 1.0, 50.0, [1000.0, 3000.0], "bandpass", fs=FS, output="sos")),
            (lambda a: a.apply_bessel_highpass(4, 300.0),
             sg.bessel(4, 300.0, "highpass", norm="mag", fs=FS, output="sos")),
        ]
        for apply, sos in cases:
            audio = aus.AudioSamples.new_mono(noise.copy(), FS)
            apply(audio)
            np.testing.assert_allclose(audio.to_numpy(), sg.sosfilt(sos, noise), atol=SCIPY_ATOL)


class TestSosFilter:
    """SosFilter streaming and scipy SOS interop."""

    def test_from_scipy_sos_streams_like_sosfilt(self, noise):
        sos = sg.ellip(6, 0.5, 60.0, 3000.0, fs=FS, output="sos")
        filt = SosFilter.from_sos(sos, FS)
        out = noise.copy()
        for start in range(0, len(out), 300):  # uneven blocks
            filt.process_block(out[start:start + 300])
        np.testing.assert_allclose(out, sg.sosfilt(sos, noise), atol=1e-9)

    def test_sos_round_trip(self):
        design = IirFilterDesign.chebyshev_ii_band("bandpass", 4, 200.0, 800.0, 40.0)
        filt = SosFilter.from_design(design, FS)
        np.testing.assert_allclose(filt.sos(), design.sos(FS), rtol=0, atol=0)

    def test_filtfilt_matches_sosfiltfilt(self, noise):
        sos = sg.butter(5, 2000.0, fs=FS, output="sos")
        out = SosFilter.from_sos(sos, FS).filtfilt(noise)
        np.testing.assert_allclose(out, sg.sosfiltfilt(sos, noise), atol=1e-9)

    def test_strided_array_is_filtered_in_place(self, noise):
        sos = sg.butter(4, 1000.0, fs=FS, output="sos")
        buf = np.zeros(2 * len(noise))
        buf[::2] = noise
        SosFilter.from_sos(sos, FS).process_block(buf[::2])
        np.testing.assert_allclose(buf[::2], sg.sosfilt(sos, noise), atol=1e-9)
        assert np.all(buf[1::2] == 0.0)

    def test_read_only_array_raises(self, noise):
        arr = noise.copy()
        arr.flags.writeable = False
        with pytest.raises(ValueError):
            SosFilter.from_sos(sg.butter(2, 1000.0, fs=FS, output="sos"), FS).process_block(arr)

    def test_degenerate_design_raises_value_error(self):
        with pytest.raises(ValueError):
            IirFilterDesign.chebyshev_i("lowpass", 2, 1000.0, 1e-17).sos(FS)

    def test_bad_sos_shape_raises(self):
        with pytest.raises(ValueError):
            SosFilter.from_sos(np.zeros((2, 5)), FS)


class TestSimpleFilters:
    """low_pass_filter / high_pass_filter / band_pass_filter."""

    @pytest.mark.parametrize("cutoff", [20.0, 1000.0, 20000.0])
    def test_match_first_order_butterworth(self, cutoff, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        lp = audio.low_pass_filter(cutoff)
        hp = audio.high_pass_filter(cutoff)
        np.testing.assert_allclose(lp.to_numpy(), sg.sosfilt(sg.butter(1, cutoff, fs=FS, output="sos"), noise), atol=SCIPY_ATOL)
        np.testing.assert_allclose(hp.to_numpy(), sg.sosfilt(sg.butter(1, cutoff, "highpass", fs=FS, output="sos"), noise), atol=SCIPY_ATOL)
        bp = audio.band_pass_filter(cutoff / 4, cutoff)
        np.testing.assert_allclose(bp.to_numpy(), sg.sosfilt(sg.butter(1, [cutoff / 4, cutoff], "bandpass", fs=FS, output="sos"), noise), atol=SCIPY_ATOL)
        # The simple filters return new objects and leave the original intact.
        np.testing.assert_array_equal(audio.to_numpy(), noise)

    def test_int16_high_pass_is_not_silent(self):
        t = np.arange(FS) / FS
        x = (10000 * np.sin(2 * np.pi * 5000.0 * t)).astype(np.int16)
        out = aus.AudioSamples.new_mono(x, FS).high_pass_filter(100.0).to_numpy()
        assert out.dtype == np.int16
        assert np.abs(out[FS // 2:]).max() > 9000  # 5 kHz passes a 100 Hz high-pass


class TestParametricEq:
    """RBJ Audio EQ Cookbook biquads."""

    @staticmethod
    def rbj(kind, f0, gain_db, q):
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * f0 / FS
        c, s = np.cos(w0), np.sin(w0)
        alpha = s / (2 * q)
        sa = 2 * np.sqrt(A) * alpha
        if kind == "peak":
            b, a = [1 + alpha * A, -2 * c, 1 - alpha * A], [1 + alpha / A, -2 * c, 1 - alpha / A]
        elif kind == "low_shelf":
            b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
            a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
            a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
        return np.array(b), np.array(a)

    @pytest.mark.parametrize("kind,method", [
        ("peak", "apply_peak_filter"),
        ("low_shelf", "apply_low_shelf"),
        ("high_shelf", "apply_high_shelf"),
    ])
    @pytest.mark.parametrize("f0,gain,q", [(100.0, 6.0, 0.7071), (1000.0, -9.0, 2.0), (8000.0, 3.0, 0.5)])
    def test_bands_match_cookbook(self, kind, method, f0, gain, q, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        getattr(audio, method)(f0, gain, q)
        b, a = self.rbj(kind, f0, gain, q)
        np.testing.assert_allclose(audio.to_numpy(), sg.lfilter(b, a, noise), atol=SCIPY_ATOL)


class TestIntegerAudio:
    """int16 audio (the default for ffmpeg-converted WAVs)."""

    def test_unity_filter_is_bit_exact(self):
        x = np.concatenate([
            (np.sin(np.arange(8000) * 0.731) * 30000).astype(np.int16),
            np.array([-32768, 32767, -1, 0, 1], dtype=np.int16),
        ])
        audio = aus.AudioSamples.new_mono(x.copy(), FS)
        audio.apply_peak_filter(1000.0, 0.0, 1.0)  # 0 dB peak: identity
        np.testing.assert_array_equal(audio.to_numpy(), x)

    def test_int16_iir_matches_scipy_within_one_lsb(self, noise):
        x = np.round(noise * 0.9 * 32767).astype(np.int16)
        audio = aus.AudioSamples.new_mono(x.copy(), FS)
        audio.apply_butterworth_highpass(4, 100.0)
        xf = np.where(x < 0, x / 32768.0, x / 32767.0)
        yf = np.clip(sg.sosfilt(sg.butter(4, 100.0, "highpass", fs=FS, output="sos"), xf), -1, 1)
        expected = np.round(np.where(yf < 0, yf * 32768.0, yf * 32767.0)).astype(np.int64)
        diff = audio.to_numpy().astype(np.int64) - expected
        assert np.abs(diff).max() <= 1
        assert abs(diff.sum()) <= 2  # no systematic bias


class TestSafety:
    """Copy, read-only and layout semantics."""

    def test_copy_is_independent(self, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        clone = audio.copy()
        clone.apply_butterworth_lowpass(4, 100.0)
        np.testing.assert_array_equal(audio.to_numpy(), noise)

    def test_read_only_input_is_never_written(self, noise):
        arr = noise.copy()
        arr.flags.writeable = False
        audio = aus.AudioSamples.new_mono(arr, FS)
        audio.apply_butterworth_lowpass(4, 100.0)
        np.testing.assert_array_equal(arr, noise)
        np.testing.assert_allclose(audio.to_numpy(), sg.sosfilt(sg.butter(4, 100.0, fs=FS, output="sos"), noise), atol=SCIPY_ATOL)

    def test_in_place_filter_writes_through_wrapped_array(self, noise):
        # Documented zero-copy behaviour: AudioSamples wraps the array.
        arr = noise.copy()
        audio = aus.AudioSamples.new_mono(arr, FS)
        audio.apply_butterworth_lowpass(2, 1000.0)
        np.testing.assert_allclose(arr, audio.to_numpy())

    def test_samples_first_array_warns(self, noise):
        frames_by_channels = np.stack([noise, noise], axis=1)  # (4096, 2), soundfile layout
        with pytest.warns(UserWarning, match="channels, samples"):
            aus.AudioSamples.new_multi(frames_by_channels.copy(), FS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aus.AudioSamples.new_multi(frames_by_channels.T.copy(), FS)
            # Under "error" the warning raises rather than being swallowed.
            with pytest.raises(UserWarning):
                aus.AudioSamples.new_multi(frames_by_channels.copy(), FS)

    def test_audio_frequency_response_is_deprecated(self, noise):
        audio = aus.AudioSamples.new_mono(noise.copy(), FS)
        with pytest.warns(DeprecationWarning):
            audio.frequency_response([100.0, 1000.0])
