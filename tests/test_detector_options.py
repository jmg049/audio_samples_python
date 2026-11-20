"""Options that shape detectors: spectral VAD, the side chain, true-peak limiting
and the onset peak-picking presets, which are given in seconds."""

import numpy as np
import pytest

import audio_samples as aus

SR = 16000


def mono(x, sr=SR):
    return aus.AudioSamples.new_mono(np.array(x, dtype=np.float64), sr)


def sine(freq, n, amp=1.0, sr=SR, phase=0.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)


def voiced(n, sr=SR):
    """f0 = 125 Hz harmonics weighted towards a 700 Hz formant."""
    t = np.arange(n) / sr
    return sum(
        0.3 / (1 + ((125 * h - 700) / 200) ** 2) * np.sin(2 * np.pi * 125 * h * t) for h in range(1, 25)
    )


def vad_config(method):
    return aus.VadConfig(
        method,
        aus.VadChannelPolicy.average_to_mono,
        frame_size=400,
        hop_size=160,
        pad_end=True,
        energy_threshold_db=-40.0,
        zcr_min=0.02,
        zcr_max=0.3,
        min_speech_frames=3,
        min_silence_frames=5,
        hangover_frames=2,
        smooth_frames=5,
        speech_band_low_hz=300.0,
        speech_band_high_hz=3400.0,
        spectral_ratio_threshold=0.6,
    )


def test_spectral_vad_rejects_loud_noise_and_accepts_voiced():
    noise = mono(0.5 * np.random.default_rng(0).uniform(-1, 1, SR // 2))
    assert all(noise.voice_activity_mask(vad_config(aus.VadMethod.energy)))
    assert not any(noise.voice_activity_mask(vad_config(aus.VadMethod.spectral)))
    assert all(mono(voiced(SR // 2)).voice_activity_mask(vad_config(aus.VadMethod.spectral)))


def test_spectral_vad_band_above_nyquist_is_an_error():
    cfg = vad_config(aus.VadMethod.spectral)
    audio = mono(sine(440, 8000, 0.5, sr=6000), sr=6000)
    with pytest.raises(ValueError):
        audio.voice_activity_mask(cfg)


def compressor(side_chain):
    return aus.CompressorConfig(
        threshold_db=-20.0,
        ratio=10.0,
        attack_ms=1.0,
        release_ms=5.0,
        makeup_gain_db=0.0,
        knee_type=aus.KneeType.hard,
        knee_width_db=0.0,
        detection_method=aus.DynamicRangeMethod.peak,
        side_chain=side_chain,
        lookahead_ms=0.0,
    )


def gain_db(before, after):
    half = len(before) // 2
    return 20 * np.log10(np.sqrt(np.mean(after[half:] ** 2)) / np.sqrt(np.mean(before[half:] ** 2)))


def compressed_gain(x, side_chain):
    audio = mono(x.copy())
    audio.apply_compressor(compressor(side_chain))
    return gain_db(x, audio.to_numpy())


def test_side_chain_high_pass_keeps_bass_from_triggering():
    bass = sine(50, 12000, 0.5)
    assert compressed_gain(bass, aus.SideChainConfig(True)) < -10
    assert compressed_gain(bass, aus.SideChainConfig(True, high_pass_freq=1000.0)) > -0.1


def test_side_chain_low_pass_keeps_treble_from_triggering():
    treble = sine(6000, 12000, 0.5)
    assert compressed_gain(treble, aus.SideChainConfig(True)) < -10
    assert compressed_gain(treble, aus.SideChainConfig(True, low_pass_freq=300.0)) > -0.1


def test_side_chain_pre_emphasis_weights_high_frequencies():
    quiet_treble = sine(7000, 12000, 10 ** (-24 / 20))  # 4 dB under the threshold
    assert compressed_gain(quiet_treble, aus.SideChainConfig(True)) == 0.0
    assert compressed_gain(quiet_treble, aus.SideChainConfig(True, pre_emphasis_db=12.0)) < -3
    quiet_bass = sine(100, 12000, 10 ** (-24 / 20))
    assert compressed_gain(quiet_bass, aus.SideChainConfig(True, pre_emphasis_db=12.0)) == 0.0


def test_side_chain_external_mix():
    assert aus.SideChainConfig(True).external_mix == 1.0
    main = sine(440, 12000, 0.05)
    key = mono(sine(1000, 12000, 0.9))

    def run(mix):
        out = mono(main.copy()).apply_compressor_sidechain(
            compressor(aus.SideChainConfig(True, external_mix=mix)), key
        )
        return gain_db(main, out.to_numpy())

    assert run(1.0) < -10
    assert run(1.0) < run(0.5) < -3
    assert run(0.0) == 0.0


def test_side_chain_mono_key_ducks_stereo():
    main = np.vstack([sine(440, 12000, 0.05)] * 2)
    stereo = aus.AudioSamples.new_multi(main.copy(), SR)
    out = stereo.apply_compressor_sidechain(compressor(aus.SideChainConfig(True)), mono(sine(1000, 12000, 0.9)))
    for c in range(2):
        assert gain_db(main[c], out.to_numpy()[c]) < -10


def test_isp_limiting_reads_true_peaks():
    # A full-scale sine at fs/4, 45 degrees off its peaks: samples at -3 dBFS, true peak 0 dBTP.
    n = 12000
    x = np.sin(np.pi / 2 * np.arange(n) + np.pi / 4)

    def limited(isp):
        cfg = aus.LimiterConfig(
            ceiling_db=-1.0,
            attack_ms=0.1,
            release_ms=50.0,
            knee_type=aus.KneeType.hard,
            knee_width_db=0.0,
            detection_method=aus.DynamicRangeMethod.peak,
            side_chain=aus.SideChainConfig(False),
            lookahead_ms=2.0,
            isp_limiting=isp,
        )
        audio = mono(x.copy())
        audio.apply_limiter(cfg)
        return audio.to_numpy()

    np.testing.assert_array_equal(limited(False), x)
    steady = limited(True)[n // 2 :]
    true_peak_db = 20 * np.log10(np.abs(steady).max() * np.sqrt(2))
    assert abs(true_peak_db + 1.0) < 0.1


@pytest.mark.parametrize(("sr", "hop"), [(16000, 160), (22050, 512), (44100, 256), (48000, 1024)])
def test_peak_picking_presets_are_in_seconds(sr, hop):
    frame = hop / sr
    for preset, separation in [
        (aus.PeakPickingConfig.music, 0.1),
        (aus.PeakPickingConfig.speech, 0.08),
        (aus.PeakPickingConfig.drums, 0.03),
    ]:
        cfg = preset(sr, hop)
        if separation >= frame:
            assert abs(cfg.min_peak_separation * frame - separation) <= frame / 2 + 1e-12
        else:
            assert cfg.min_peak_separation == 1


def test_onset_presets_take_the_sample_rate():
    for rate in (22050, 44100):
        c = aus.OnsetDetectionConfig.percussive(rate)
        assert c.peak_picking.min_peak_separation == aus.PeakPickingConfig.drums(rate, c.hop_size).min_peak_separation
        assert aus.SpectralFluxConfig.musical(rate).peak_picking.min_peak_separation == round(0.1 * rate / 512)
        aus.ComplexOnsetConfig.speech(rate)
