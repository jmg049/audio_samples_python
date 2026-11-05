"""``audio_samples.mpl`` matrix and waveform displays: axis types, value scales, formatters,
colormaps, colorbars and panels. Coordinates are checked against the frequencies each axis
type promises, using the Agg backend."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import audio_samples as aus  # noqa: E402
from audio_samples import mpl  # noqa: E402
from audio_samples import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


DATA = np.abs(np.random.default_rng(1).normal(size=(64, 40))) + 1e-3


@pytest.mark.parametrize("axis", [a for a in mpl.AXIS_TYPES if a not in ("vqt_hz", "vqt_note", "vqt_oct3", "vqt_fjs", "chroma_fjs")])
def test_every_axis_type_draws(axis):
    fig, ax, mesh = mpl.plot_matrix(DATA, x_axis="time", y_axis=axis, Sa=261.63, mela=1)
    fig.canvas.draw()
    assert mesh is ax.collections[0]
    fig, ax, _ = mpl.plot_matrix(DATA.T, x_axis=axis, y_axis="frames", Sa=261.63, mela=1)
    fig.canvas.draw()


@pytest.mark.parametrize("axis", ["vqt_hz", "vqt_note", "vqt_oct3", "vqt_fjs", "chroma_fjs"])
def test_tuning_axes_draw(axis):
    fig, ax, _ = mpl.plot_matrix(DATA[:24], y_axis=axis, intervals="ji5", bins_per_octave=12, unison="C", fmin=55.0)
    fig.canvas.draw()
    with pytest.raises(ValueError):
        mpl.plot_matrix(DATA[:24], y_axis=axis)


def test_unknown_axis_and_shape():
    with pytest.raises(ValueError):
        mpl.plot_matrix(DATA, y_axis="nope")
    with pytest.raises(ValueError):
        mpl.plot_matrix(DATA[0])


def _edges(mesh):
    return np.asarray(mesh.get_coordinates())


def test_coordinates_follow_the_axis():
    _, _, mesh = mpl.plot_matrix(DATA, x_axis="time", y_axis="linear", sample_rate=8000, hop_length=100, n_fft=126)
    coords = _edges(mesh)
    assert np.isclose(coords[0, 1, 0] - coords[0, 0, 0], 100 / 8000)
    assert np.isclose(coords[1, 0, 1] - coords[0, 0, 1], 8000 / 126)
    _, _, mesh = mpl.plot_matrix(DATA, y_axis="cqt_hz", fmin=55.0, bins_per_octave=24)
    y = _edges(mesh)[:, 0, 1]
    assert np.isclose(y[25] / y[1], 2.0)
    fig, ax, _ = mpl.plot_matrix(DATA, x_coords=np.arange(41.0), y_coords=np.arange(65.0))
    assert ax.get_xlim()[1] == pytest.approx(40.0)


def test_value_scales():
    z = DATA * np.exp(1j * np.random.default_rng(2).uniform(-3, 3, DATA.shape))
    for scale in ("dB", "dB[0.1]", "dB[power]", "dB[power,0.1]", "dBFS", "dBFS[power]", "phase"):
        fig, ax, mesh = mpl.plot_matrix(z, y_axis="linear", value_scale=scale, colorbar=True)
        fig.canvas.draw()
    _, _, mesh = mpl.plot_matrix(DATA, value_scale="dB[power]", top_db=10.0)
    values = np.asarray(mesh.get_array())
    assert values.max() - values.min() <= 10.0 + 1e-9
    assert np.isclose(values.max(), 10 * np.log10(DATA.max()))
    _, _, mesh = mpl.plot_matrix(DATA, value_scale="dBFS")
    assert np.isclose(np.asarray(mesh.get_array()).max(), 0.0)
    for scale in ("dphase", "dphase_t"):
        _, _, mesh = mpl.plot_matrix(z, x_axis="time", y_axis="hz", value_scale=scale)
        values = np.asarray(mesh.get_array())
        assert np.abs(values).max() <= np.pi + 1e-9
    for bad, data in (("nope", DATA), ("phase", DATA), (None, z), ("dBFS[0.1]", DATA)):
        with pytest.raises(ValueError):
            mpl.plot_matrix(data, value_scale=bad)
    with pytest.raises(ValueError):
        mpl.plot_matrix(z, value_scale="dphase")


def test_phase_advance_of_a_stationary_tone_is_zero():
    sr, hop = 8000, 100
    f0 = 400.0
    times = np.arange(20) * hop / sr
    z = np.exp(2j * np.pi * f0 * times)[None, :]
    _, _, mesh = mpl.plot_matrix(z, x_coords=times, y_coords=np.array([f0]), value_scale="dphase")
    assert np.abs(np.asarray(mesh.get_array())).max() < 1e-9


def test_infer_colormap():
    assert mpl.infer_colormap(np.array([True, False])).name == "gray_r"
    assert mpl.infer_colormap(np.array([0.0, 1.0])).name == "magma"
    assert mpl.infer_colormap(np.linspace(-1, 1, 100)).name == "coolwarm"
    # One outlier does not make data two-signed unless asked.
    data = np.r_[np.ones(1000), -1.0]
    assert mpl.infer_colormap(data).name == "magma"
    assert mpl.infer_colormap(data, robust=False).name == "coolwarm"
    assert mpl.infer_colormap(np.array([1.0, 3.0]), threshold=2.0, diverging="bwr").name == "bwr"
    assert mpl.infer_colormap(np.array([np.nan])).name == "magma"


def _format(formatter, x, lim=(1.0, 1000.0)):
    fig, ax = plt.subplots()
    ax.set_xlim(*lim)
    ax.plot(list(lim), [0, 0])
    ax.xaxis.set_major_formatter(formatter)
    return formatter(x)


def test_formatters():
    assert _format(mpl.TimeFormatter(unit="ms"), 0.5, (0, 10)) == "500"
    assert _format(mpl.TimeFormatter(unit="m"), 120.0, (0, 300)) == "2"
    assert _format(mpl.TimeFormatter(unit="h"), 3600.0, (0, 7200)) == "1"
    assert _format(mpl.TimeFormatter(), 75.0, (0, 300)) == "1:15"
    assert _format(mpl.TimeFormatter(), 3700.0, (0, 7200)) == "1:01:40"
    assert _format(mpl.TimeFormatter(lag=True, unit="s"), 8.0, (0, 10)) == "-2"
    with pytest.raises(ValueError):
        mpl.TimeFormatter(unit="d")
    assert _format(mpl.NoteFormatter(unicode=False), 440.0) == "A4"
    assert _format(mpl.NoteFormatter(octave=False, unicode=False), 440.0) == "A"
    assert _format(mpl.NoteFormatter(key="F:maj", unicode=False), 466.16) == "Bb4"
    assert _format(mpl.LogHzFormatter(), 440.0) == "440"
    assert _format(mpl.LogHzFormatter(major=False), 440.0) == ""
    assert _format(mpl.LogHzFormatter(major=False), 440.0, (400, 500)) == "440"
    assert _format(mpl.EngFormatter(unit="Hz"), 2000.0) == "2 kHz"
    assert _format(mpl.SvaraFormatter(Sa=261.63, abbr=True), 392.0) == "P"
    assert _format(mpl.SvaraFormatter(Sa=261.63, abbr=True, mela=1, unicode=False), 293.66) == "G1"
    assert _format(mpl.ChromaFormatter(unicode=False), 9.0, (0, 12)) == "A"
    assert _format(mpl.ChromaSvaraFormatter(unicode=False), 7.0, (0, 12)) == "P"
    assert _format(mpl.ChromaSvaraFormatter(mela=1, unicode=False), 2.0, (0, 12)) == "G1"
    assert _format(mpl.ChromaFJSFormatter(intervals=[1.0, 1.25, 1.5], unicode=False), 1.0, (0, 3)) == "E^5"
    with pytest.raises(ValueError):
        mpl.ChromaFJSFormatter(intervals="ji5")
    assert _format(mpl.ChromaFJSFormatter(intervals="pythagorean", bins_per_octave=12), 7.0, (0, 12))
    assert _format(mpl.TonnetzFormatter(), 0.0, (0, 5)) == "5$_x$"
    assert _format(mpl.TonnetzFormatter(), 9.0, (0, 5)) == ""
    f = mpl.FJSFormatter(fmin=110.0, n_bins=24, bins_per_octave=12, intervals="ji5", unison="A", unicode=False)
    assert _format(f, 165.0, (110, 440)) == "E"


def test_adaptive_waveform_follows_the_view():
    y = np.sin(np.linspace(0, 400, 44100))
    fig, ax, w = mpl.plot_waveform_adaptive(y, sample_rate=22050, label="a")
    assert w.envelope.get_visible() and not w.steps.get_visible()
    ax.set_xlim(0.0, 0.1)
    assert w.steps.get_visible() and not w.envelope.get_visible()
    assert w.steps.get_xdata().size <= 0.1 * 22050 + 3
    ax.set_xlim(0.0, 2.0)
    assert w.envelope.get_visible()
    w.disconnect()
    ax.set_xlim(0.0, 0.1)
    assert w.envelope.get_visible()
    fig.canvas.draw()
    audio = aus.AudioSamples.new_mono(y.astype(np.float32), 22050)
    _, ax, w = mpl.plot_waveform_adaptive(audio, offset=1.0, invert=True, invert_color="r")
    assert ax.get_xlim()[0] == 1.0
    _, ax, w = mpl.plot_waveform_adaptive(y, sample_rate=22050, transpose=True, mask=y > 0)
    assert ax.get_ylim()[1] == pytest.approx(2.0)
    assert np.isnan(w.samples).any()
    _, _, w = mpl.plot_waveform_adaptive(np.stack([y, -y]), sample_rate=22050)
    assert np.abs(w.samples).max() == 0.0
    with pytest.raises(ValueError):
        mpl.plot_waveform_adaptive(y)
    with pytest.raises(ValueError):
        mpl.plot_waveform_adaptive(y, sample_rate=22050, max_points=0)


def test_bars():
    y = np.sin(np.linspace(0, 400, 22050))
    fig, ax, bars = mpl.plot_waveform_bars(y, sample_rate=22050, n_bars=20)
    assert len(bars.get_paths()) == 20
    fig.canvas.draw()
    _, _, bars = mpl.plot_waveform_bars(y, sample_rate=22050, n_bars=20, invert=True, invert_color="k")
    assert len(bars.get_paths()) == 40
    _, ax, _ = mpl.plot_waveform_bars(y, sample_rate=22050, n_bars=10, transpose=True, offset=0.5)
    assert ax.get_ylim()[0] == 0.5
    for bad in (dict(n_bars=0), dict(gap_ratio=1.0), dict(rounding_ratio=2.0)):
        with pytest.raises(ValueError):
            mpl.plot_waveform_bars(y, sample_rate=22050, **bad)


def test_f0_transform_and_plot():
    t = mpl.F0Transform(np.array([220.0, 220.0, 440.0]), sample_rate=100, hop_length=10, scale=1.0)
    out = t.transform_non_affine(np.array([[0.0, 0.0], [0.0, 1.0], [0.2, 0.0]]))
    np.testing.assert_allclose(out, [[0.0, 220.0], [0.0, 440.0], [0.2, 440.0]])
    tt = mpl.F0Transform(np.array([100.0, 100.0]), sample_rate=100, hop_length=10, transpose=True, scale=1.0)
    np.testing.assert_allclose(tt.transform_non_affine(np.array([[1.0, 0.05]])), [[200.0, 0.05]])
    y = np.sin(np.linspace(0, 400, 22050))
    f0 = np.linspace(200.0, 400.0, 44)
    fig, ax, _ = mpl.plot_waveform_f0(y, f0=f0, sample_rate=22050)
    fig.canvas.draw()
    assert ax.get_ylim()[0] < 200.0 < 400.0 < ax.get_ylim()[1]
    fig, _, _ = mpl.plot_waveform_f0(y, f0=f0, sample_rate=22050, method="bars", freq_axis="cqt_svara", Sa=220.0)
    fig.canvas.draw()
    with pytest.raises(ValueError):
        mpl.plot_waveform_f0(y, f0=f0, sample_rate=22050, method="nope")


def test_colorbars():
    fig, ax, mesh = mpl.plot_matrix(DATA, value_scale="dB")
    bar = mpl.add_db_colorbar(mesh)
    assert bar.ax.get_ylabel() == "dB"
    z = np.exp(1j * DATA)
    fig, ax, mesh = mpl.plot_matrix(z, value_scale="phase")
    bar = mpl.add_phase_colorbar(mesh, numticks=5)
    fig.canvas.draw()
    labels = [t.get_text() for t in bar.ax.get_yticklabels()]
    assert labels == ["-π", "-π/2", "0", "π/2", "π"]
    with pytest.raises(ValueError):
        mpl.add_phase_colorbar(mesh, numticks=1)


def test_grid_and_legend():
    y = np.sin(np.linspace(0, 100, 2205))
    axes = mpl.plot_grid("waveform", y, -y, sample_rate=22050, labels=["a", "b"], titles=["A", "B"])
    assert axes.shape == (2, 1) and axes[0, 0].get_title() == "A"
    legend = mpl.legend_for_axes(axes)
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]
    axes = mpl.plot_grid("bars", np.stack([y, y]), sample_rate=22050, orient="h", n_bars=[4, 8])
    assert axes.shape == (1, 2)
    assert len(axes[0, 1].collections[0].get_paths()) == 8
    axes = mpl.plot_grid("matrix", DATA, DATA * 2, shared=True, y_axis="linear")
    a, b = (ax.collections[0].get_clim() for ax in axes.ravel())
    assert a == b
    from cycler import cycler

    axes = mpl.plot_grid("waveform", y, y, sample_rate=22050, prop_cycle=cycler(color=["r", "g"]))
    assert axes[1, 0].lines[0].get_color() == "g"
    for bad in (lambda: mpl.plot_grid("nope", y), lambda: mpl.plot_grid("bars", y, y, orient="z", sample_rate=1)):
        with pytest.raises(ValueError):
            bad()


def test_highlight():
    fig, ax = plt.subplots()
    ax.set_facecolor("black")
    line = ax.plot([0, 1])[0]
    effects = mpl.highlight_effects(artist=line)
    assert line.get_path_effects() == effects
    assert effects[0]._gc["foreground"] == "white"
    ax.set_facecolor("white")
    assert mpl.highlight_effects(ax=ax)[0]._gc["foreground"] == "black"
    fig, ax, _ = mpl.plot_matrix(np.zeros((4, 4)) + 1e-9, cmap="gray", vmin=0.0, vmax=1.0)
    assert mpl.highlight_effects(ax=ax)[0]._gc["foreground"] == "white"
    assert mpl.highlight_effects(color="red")[0]._gc["foreground"] == "red"
    assert U.key_to_notes("C:maj") is not None
