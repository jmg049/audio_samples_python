"""Facts about each public librosa 1.0 function, for ``audio_samples.migrate``.

``TABLE`` maps a public librosa name to what replaces it. The same facts are
written out for people in ``docs/source/librosa_migration.rst``. The test suite
checks this table against the document, against the installed librosa (every
public function appears, every parameter and default is accounted for) and
against ``audio_samples`` (every equivalent exists).

A ``params`` string lists the parameters of the librosa function:
``"y | n_fft hop_length"`` means ``y`` can be positional and the others are
keyword-only, ``*signals`` means any number of positional arguments, and ``**``
means the function takes more keywords than it names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence, Union

EQUIVALENT = "equivalent"
WITH_DIFFERENCES = "with differences"
NOT_PROVIDED = "not provided"


@dataclass(frozen=True)
class Renamed:
    """The keyword has another name in the equivalent."""

    to: str
    note: str = ""


@dataclass(frozen=True)
class Absorbed:
    """The argument is not needed in the equivalent."""

    reason: str


@dataclass(frozen=True)
class Unsupported:
    """The equivalent has no such option; passing it at all needs another approach."""

    reason: str


@dataclass(frozen=True)
class Only:
    """Only these values of the option are supported."""

    values: tuple
    reason: str = ""
    # Accepted spellings that are not constants, such as ``np.float64``.
    names: tuple[str, ...] = ()

    def allows(self, value: object) -> bool:
        return any(value == allowed and type(value) is type(allowed) for allowed in self.values)

    def describe(self) -> str:
        return " or ".join([*(repr(value) for value in self.values), *self.names])


@dataclass(frozen=True)
class Default:
    """The default differs. It matters only when the call does not pass the keyword."""

    librosa: object
    ours: str
    advice: str = ""
    value_note: str = ""


@dataclass(frozen=True)
class Same:
    """The default is written differently and computes the same thing, so nothing to report."""

    reason: str


class _Named:
    """A default that is a function or a sentinel, shown by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


Option = Union[Renamed, Absorbed, Unsupported, Only, Default, Same]


@dataclass(frozen=True)
class Form:
    """Another entry point, used when the call passes one of the ``when`` keywords.

    Its ``options`` replace those of the entry.
    """

    when: tuple[str, ...]
    ours: tuple[str, ...]
    options: Mapping[str, Option] = field(default_factory=dict)
    match: str | None = None
    note: str | None = None
    shape: str | None = None
    behaviour: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    name: str
    positional: tuple[str, ...]
    parameters: frozenset[str]
    var_positional: bool
    takes_keywords: bool
    ours: tuple[str, ...]
    match: str
    note: str = ""
    shape: str | None = None
    options: Mapping[str, Option] = field(default_factory=dict)
    forms: tuple[Form, ...] = ()
    behaviour: tuple[str, ...] = ()


TABLE: dict[str, Entry] = {}


def _parse_params(params: str) -> tuple[tuple[str, ...], frozenset[str], bool, bool]:
    if "|" in params:
        head, _, tail = params.partition("|")
    else:
        # Without a bar, every named parameter can be positional.
        head, tail = params, ""
    positional = []
    var_positional = False
    keywords = []
    for name in head.split():
        if name == "**":
            keywords.append(name)
        elif name.startswith("*"):
            var_positional = True
            positional.append(name[1:])
        else:
            positional.append(name)
    keywords += tail.split()
    takes_keywords = "**" in keywords
    named = [k for k in keywords if k != "**"]
    return tuple(positional), frozenset([*positional, *named]), var_positional, takes_keywords


def _e(
    name: str,
    params: str,
    ours: Sequence[str] | str,
    match: str,
    note: str = "",
    *,
    shape: str | None = None,
    options: Mapping[str, Option] | None = None,
    forms: Sequence[Form] = (),
    behaviour: Sequence[str] = (),
) -> None:
    positional, parameters, var_positional, takes_keywords = _parse_params(params)
    TABLE[name] = Entry(
        name=name,
        positional=positional,
        parameters=parameters,
        var_positional=var_positional,
        takes_keywords=takes_keywords,
        ours=(ours,) if isinstance(ours, str) else tuple(ours),
        match=match,
        note=note,
        shape=shape,
        options=dict(options or {}),
        forms=tuple(forms),
        behaviour=tuple(behaviour),
    )


def _none(reason: str, *names_and_params: str) -> None:
    """Functions with no counterpart. Each argument is ``"name(params)"``."""
    for item in names_and_params:
        name, _, params = item.partition("(")
        _e(name, params.rstrip(")"), (), NOT_PROVIDED, reason)


def _as(name: str, target: str) -> None:
    """A second public name for the same function."""
    TABLE[name] = replace(TABLE[target], name=name)


# Reused text ---------------------------------------------------------------

_Y = Absorbed("call the method on an AudioSamples, which holds the samples and the sample rate.")
_SR = _Y
_NOT_METHOD = Absorbed("not used by this method.")
_LAYOUT = Absorbed("an AudioSamples knows its layout.")
_AUDIO: dict[str, Option] = {"y": _Y, "sr": _SR}
_DTYPE = Unsupported("the output precision follows the sample type; convert the audio with as_f32, as_f64 or to_format first.")
_OUT = Unsupported("the array forms take out=: forms.stft, forms.istft, forms.to_mono, forms.to_stereo and forms.to_multi.")
_ARRAY_SR = Default(22050, "a required sr", "Pass sr=...")
_ARRAY_HOP = Default(512, "a required hop_length", "Pass hop_length=...")
_HOP_NONE = Default(512, "hop_length=None", "Pass hop_length=512 with frames=.")
_INFER_N_FFT = Same("None derives the FFT size from the number of bins.")
_FRAMES = "(channels, n_features, n_frames), also for mono audio; index [0] for librosa's mono shape."
_SCALARS = "(channels, 1, n_frames), also for mono audio; index [0] for librosa's mono shape."
_TRACK = "(channels, n_frames), also for mono audio; index [0] for librosa's mono shape."
_MEL_DTYPE = Unsupported("the mel basis is computed in float64 and rounded once to the output type.")
_NOT_AUDIO = Absorbed("this form takes a spectrogram rather than audio.")
_AUDIO_ONLY_KW = "frames were computed elsewhere."
_STFT_KWS = ("n_fft", "hop_length", "win_length", "window", "center", "pad_mode")
_STFT_PARAMS = " ".join(_STFT_KWS)


def _free(*names: str) -> dict[str, Option]:
    return {name: _ARRAY_SR if name == "sr" else _ARRAY_HOP for name in names}


def _computed_elsewhere(*names: str) -> dict[str, Option]:
    return {name: Absorbed(_AUDIO_ONLY_KW) for name in names}


# Core: audio ---------------------------------------------------------------

_e(
    "librosa.load",
    "path | sr mono offset duration dtype res_type",
    ("io.read", "io.read_and_resample"),
    WITH_DIFFERENCES,
    "WAV and FLAC only (librosa: every libsndfile format, also MP3 and Ogg); a path or a binary file object.",
    options={
        "path": Renamed("fp", "A path to a WAV or FLAC file, or a binary file object holding one."),
        "sr": Default(
            22050,
            "the file's own rate",
            "librosa resamples to 22050 Hz here; io.read does not. Use io.read_and_resample(fp, 22050) to do the same.",
            value_note="io.read keeps the file's rate. For a number use io.read_and_resample(fp, rate); for None use io.read.",
        ),
        "mono": Default(
            True,
            "the file's own channels",
            "librosa mixes to mono here; io.read does not. Call .to_mono() on the result to do the same.",
            value_note="io.read keeps the file's channels. Call .to_mono() on the result for mono.",
        ),
        "offset": Unsupported("use forms.read_segment(path, offset=..., duration=...)."),
        "duration": Unsupported("use forms.read_segment(path, offset=..., duration=...)."),
        "dtype": Default(
            _Named("numpy.float32"),
            "the file's own sample type",
            "librosa converts to float32 here; io.read does not. Use io.read(fp, as_type=SampleType.F32) to do the same.",
            value_note="io.read keeps the file's sample type unless as_type= is given.",
        ),
        "res_type": Renamed("quality", "An argument of io.read_and_resample; see the guide's section on resampling."),
    },
    behaviour=("Returns an AudioSamples holding samples and rate, not a (y, sr) tuple.",),
)
_e(
    "librosa.loadx",
    "key | hq **",
    "generation.example_clip",
    WITH_DIFFERENCES,
    "Returns an AudioSamples holding a synthesised clip rather than a (y, sr) tuple of a recording; there is nothing to download.",
    options={
        "key": Renamed("name", "Clip names differ; see generation.example_clip_names()."),
        "hq": Absorbed("clips are rendered at the rate asked for."),
        "sr": Renamed("sample_rate"),
        "duration": Renamed("duration_secs"),
        "dtype": Renamed("dtype", "The default is float32, as here."),
    },
)
_e(
    "librosa.stream",
    "path | block_length frame_length hop_length sr mono offset duration fill_value res_type dtype",
    "io.StreamedAudioReader",
    WITH_DIFFERENCES,
    "A block reader over WAV and FLAC with its own interface: windows(frame_length + (block_length - 1) * hop_length, "
    "block_length * hop_length) gives the same blocks.",
    options={
        "path": Renamed("fp"),
        "block_length": Absorbed("it goes into the size given to windows()."),
        "frame_length": Absorbed("it goes into the size given to windows()."),
        "hop_length": Absorbed("it goes into the size given to windows()."),
        "sr": Unsupported("use forms.stream_blocks(path, ..., sr=...)."),
        "mono": Default(True, "the file's own channels", "librosa mixes each block to mono; the reader does not."),
        "offset": Unsupported("use forms.stream_blocks(path, ..., offset=...)."),
        "duration": Unsupported("use forms.stream_blocks(path, ..., duration=...)."),
        "fill_value": Unsupported("use forms.stream_blocks(path, ..., fill_value=...)."),
        "res_type": Unsupported("use forms.stream_blocks(path, ..., res_type=...)."),
        "dtype": Default(_Named("numpy.float32"), "the file's own sample type"),
    },
)
_e(
    "librosa.get_samplerate",
    "path",
    "io.info",
    WITH_DIFFERENCES,
    "io.info(path).sample_rate; for loaded audio AudioSamples.sample_rate.",
    options={"path": Renamed("fp")},
)
_e(
    "librosa.get_duration",
    "| y sr S n_fft hop_length center path",
    "AudioSamples.duration_seconds",
    WITH_DIFFERENCES,
    "Audio only; no S= or path= forms.",
    options={
        **_AUDIO,
        "S": Unsupported("use forms.get_duration(S=...)."),
        "n_fft": Unsupported("use forms.get_duration(S=..., n_fft=...)."),
        "hop_length": Unsupported("use forms.get_duration(S=..., hop_length=...)."),
        "center": Unsupported("use forms.get_duration(S=..., center=...)."),
        "path": Unsupported("use forms.get_duration(path=...)."),
    },
)
_e(
    "librosa.resample",
    "y | orig_sr target_sr res_type fix scale axis **",
    ("AudioSamples.resample", "AudioSamples.resample_by_ratio"),
    WITH_DIFFERENCES,
    "Own windowed-sinc resampler, not soxr; the output length is exactly ceil(n * target / orig).",
    options={
        "y": _Y,
        "orig_sr": _SR,
        "target_sr": Renamed("target_sample_rate"),
        "res_type": Default(
            "soxr_hq",
            'quality="fast"',
            'Pass quality="high" for the closest result.',
            value_note='becomes quality=; "high" is closest to soxr_hq.',
        ),
        "fix": Only((True,), "The length is always exact; forms.resample takes fix."),
        "scale": Only((False,), "Use forms.resample(y, ..., scale=True)."),
        "axis": _LAYOUT,
    },
)
_e(
    "librosa.to_mono",
    "*signals | pad norm out",
    "AudioSamples.to_mono",
    EQUIVALENT,
    "The default method is the channel average.",
    options={
        "signals": _Y,
        "pad": Unsupported("use forms.to_mono(*signals, pad=...)."),
        "norm": Only((True,), "Use forms.to_mono(*signals, norm=False) for the sum."),
        "out": _OUT,
    },
)
_e(
    "librosa.to_stereo",
    "| left right downmix pad norm out",
    "AudioSamples.to_stereo",
    WITH_DIFFERENCES,
    "Takes a StereoConversionMethod; for two separate signals use AudioSamples.stack.",
    options={
        "left": Unsupported("use forms.to_stereo(left=..., right=...)."),
        "right": Unsupported("use forms.to_stereo(left=..., right=...)."),
        "downmix": Unsupported("use forms.to_stereo(left=..., right=..., downmix=...)."),
        "pad": Unsupported("use forms.to_stereo(left=..., right=..., pad=...)."),
        "norm": Unsupported("use forms.to_stereo(left=..., right=..., norm=...)."),
        "out": _OUT,
    },
)
_e(
    "librosa.to_multi",
    "*signals | downmix pad norm out",
    "AudioSamples.stack",
    WITH_DIFFERENCES,
    "Takes a list of AudioSamples; no pad or trim to a common length.",
    options={
        "signals": Renamed("sources", "Pass the signals as one list of AudioSamples."),
        "downmix": Unsupported("use forms.to_multi(*signals, downmix=...)."),
        "pad": Unsupported("use forms.to_multi(*signals, pad=...)."),
        "norm": Unsupported("use forms.to_multi(*signals, norm=...)."),
        "out": _OUT,
    },
)
_e(
    "librosa.autocorrelate",
    "y | max_size axis",
    "AudioSamples.autocorrelation",
    WITH_DIFFERENCES,
    "First channel only; unbiased estimate: r[k] * (n - k) equals librosa's raw sum.",
    options={"y": _Y, "max_size": Renamed("max_lag"), "axis": _LAYOUT},
)
_e(
    "librosa.lpc",
    "y | order axis",
    ("AudioSamples.lpc", "effects.lpc"),
    EQUIVALENT,
    "Burg's method; one row per channel.",
    options={"y": _Y, "axis": Absorbed("the method returns one row per channel; effects.lpc takes axis=.")},
    shape="(channels, order + 1), also for mono audio; index [0] for librosa's mono shape.",
)
_e(
    "librosa.zero_crossings",
    "y | threshold ref_magnitude pad zero_pos axis",
    "AudioSamples.zero_crossings",
    WITH_DIFFERENCES,
    "Returns a count, not a boolean mask; AudioSamples.zero_crossing_rate_frames gives the frame-wise rate.",
    options={
        "y": _Y,
        "threshold": Unsupported("use arrays.zero_crossings(y, threshold=...)."),
        "ref_magnitude": Unsupported("use arrays.zero_crossings(y, ref_magnitude=...)."),
        "pad": Unsupported("use arrays.zero_crossings(y, pad=...)."),
        "zero_pos": Unsupported("use arrays.zero_crossings(y, zero_pos=...)."),
        "axis": _LAYOUT,
    },
)
_e(
    "librosa.clicks",
    "| times frames sr hop_length click_freq click_duration click length",
    "effects.clicks",
    WITH_DIFFERENCES,
    "float64 output unless dtype=numpy.float32 (librosa's default).",
    options={"sr": _ARRAY_SR, "hop_length": _HOP_NONE},
)
_e(
    "librosa.tone",
    "frequency | sr length duration phi",
    ("generation.sine_wave", "generation.cosine_wave"),
    WITH_DIFFERENCES,
    "No phi; returns an AudioSamples.",
    options={
        "sr": Renamed("sample_rate", "The default is 44100, not 22050."),
        "length": Renamed("duration_secs", "Give the length in seconds: length / sr."),
        "duration": Renamed("duration_secs"),
        "phi": Unsupported("use arrays.tone(frequency, ..., phi=...)."),
    },
)
_e(
    "librosa.chirp",
    "| fmin fmax sr length duration linear phi",
    ("generation.exponential_chirp", "generation.chirp"),
    WITH_DIFFERENCES,
    "generation.exponential_chirp for linear=False (librosa's default), generation.chirp for linear=True. No phi; returns an AudioSamples.",
    options={
        "fmin": Renamed("start_freq", "The first argument of generation.chirp is f0."),
        "fmax": Renamed("end_freq", "The second argument of generation.chirp is f1."),
        "sr": Renamed("sample_rate", "The default is 44100, not 22050."),
        "length": Renamed("duration_secs", "Give the length in seconds: length / sr."),
        "duration": Renamed("duration_secs"),
        "linear": Absorbed("choose the function instead."),
        "phi": Unsupported("use arrays.chirp(fmin=..., fmax=..., phi=...)."),
    },
)
_e(
    "librosa.mu_compress",
    "x | mu quantize",
    ("audio_samples.mu_law_encode", "AudioSamples.mu_compress"),
    EQUIVALENT,
    "mu_law_encode returns librosa's default integer codes; AudioSamples.mu_compress is quantize=False.",
    options={"quantize": Absorbed("choose the function instead: mu_law_encode for True, AudioSamples.mu_compress for False.")},
)
_e(
    "librosa.mu_expand",
    "x | mu quantize",
    ("audio_samples.mu_law_decode", "AudioSamples.mu_expand"),
    EQUIVALENT,
    "As mu_compress.",
    options={
        "x": Renamed("codes"),
        "quantize": Absorbed("choose the function instead: mu_law_decode for True, AudioSamples.mu_expand for False."),
    },
)

# Core: spectral representations --------------------------------------------

_e(
    "librosa.stft",
    "y | n_fft hop_length win_length window center dtype pad_mode out",
    ("AudioSamples.stft_matrix", "spectral.stft"),
    EQUIVALENT,
    "All pad_mode values, win_length, any window. AudioSamples.spectrogram_matrix gives abs(stft) ** power.",
    options={"y": _Y, "dtype": _DTYPE, "out": _OUT},
    shape="(channels, 1 + n_fft // 2, n_frames), also for mono audio; index [0] for librosa's mono shape. spectral.stft keeps librosa's shape.",
)
_e(
    "librosa.istft",
    "stft_matrix | hop_length win_length n_fft window center dtype length out",
    ("AudioSamples.istft_matrix", "spectral.istft"),
    WITH_DIFFERENCES,
    "A bin count that does not match n_fft and length=0 raise. AudioSamples.istft_matrix is a static method and needs sample_rate=.",
    options={"out": _OUT},
    behaviour=("AudioSamples.istft_matrix returns an AudioSamples; spectral.istft returns an array.",),
)
_e(
    "librosa.magphase",
    "D | power",
    "AudioSamples.magphase",
    WITH_DIFFERENCES,
    "A static method; 2-D complex64 or complex128 input.",
    options={
        "D": Renamed("complex_spect"),
        "power": Same("None means 1 here."),
    },
)
_e(
    "librosa.reassigned_spectrogram",
    "y | sr S n_fft hop_length win_length window center reassign_frequencies reassign_times ref_power fill_nan clip dtype pad_mode",
    "AudioSamples.reassigned_spectrogram",
    EQUIVALENT,
    options={**_AUDIO, "S": Unsupported("use forms.reassigned_spectrogram(y, S=...)."), "dtype": _DTYPE},
    shape="Each array is (channels, 1 + n_fft // 2, n_frames), also for mono audio; index [0] for librosa's mono shape.",
)
_e(
    "librosa.phase_vocoder",
    "D | rate t_out kind hop_length n_fft",
    "effects.phase_vocoder",
    EQUIVALENT,
    "The deprecated hop_length and n_fft are accepted with a FutureWarning and unused, as in librosa 1.0.",
)
_e(
    "librosa.griffinlim",
    "S | n_iter hop_length win_length n_fft window center dtype length pad_mode momentum init rng random_state",
    ("effects.griffinlim", "AudioSamples.griffinlim_audio"),
    WITH_DIFFERENCES,
    "init=None matches; random initial phases come from this library's RNG (integer seeds only).",
    options={"rng": Default(None, "rng=0", "Repeated calls give the same output; pass rng=None for a new seed each call.")},
)
_e(
    "librosa.iirt",
    "y | sr win_length hop_length center tuning pad_mode flayout res_type **",
    ("AudioSamples.iirt", "pitch_filterbank.iirt"),
    WITH_DIFFERENCES,
    "The two lower sample rates use this library's resampler rather than soxr.",
    options={"y": _Y, "sr": _SR},
    forms=(
        Form(
            when=("sample_rates",),
            ours=("pitch_filterbank.iirt",),
            options={"sr": _ARRAY_SR},
        ),
    ),
)
_e(
    "librosa.fmt",
    "y | t_min n_fmt kind beta over_sample axis",
    "spectral.mellin_transform",
    EQUIVALENT,
    "Named mellin_transform. float32 input stays in single precision.",
)
_e(
    "librosa.pcen",
    "S | sr hop_length gain bias power time_constant eps b max_size ref axis max_axis zi return_zf",
    "features.pcen",
    EQUIVALENT,
    "zi and return_zf give streaming state.",
    options=_free("sr", "hop_length"),
)
_TOP_DB = Default(80.0, "top_db=None", "librosa applies an 80 dB floor here; pass top_db=80.0 to do the same.")
_REF_NOTE = 'ref as a number, "max" or numpy.max; per-channel reference.'
_e("librosa.amplitude_to_db", "S | ref amin top_db axes", "spectral.amplitude_to_db", EQUIVALENT, _REF_NOTE, options={"top_db": _TOP_DB})
_e("librosa.power_to_db", "S | ref amin top_db axes", "spectral.power_to_db", EQUIVALENT, _REF_NOTE, options={"top_db": _TOP_DB})
_e("librosa.db_to_amplitude", "S_db | ref", "spectral.db_to_amplitude", EQUIVALENT)
_e("librosa.db_to_power", "S_db | ref", "spectral.db_to_power", EQUIVALENT)
_e(
    "librosa.perceptual_weighting",
    "S frequencies | kind **",
    "features.perceptual_weighting",
    EQUIVALENT,
    behaviour=("top_db defaults to None here (librosa: 80, through **kwargs); pass top_db=80.0 to do the same.",),
)
_MIN_DB_BEHAVIOUR = ("min_db defaults to None here (librosa: -80, through **kwargs); pass min_db=-80.0 to do the same.",)
_e("librosa.frequency_weighting", "frequencies | kind **", "features.frequency_weighting", EQUIVALENT, behaviour=_MIN_DB_BEHAVIOUR)
_e("librosa.multi_frequency_weighting", "frequencies | kinds **", "features.multi_frequency_weighting", EQUIVALENT, behaviour=_MIN_DB_BEHAVIOUR)
for _letter in "ABCD":
    _e(f"librosa.{_letter}_weighting", "frequencies | min_db", f"features.{_letter.lower()}_weighting", EQUIVALENT, "Lower-case name.")
_e(
    "librosa.Z_weighting",
    "frequencies | min_db",
    "features.z_weighting",
    EQUIVALENT,
    "Lower-case name.",
    options={"min_db": Default(None, "min_db=-80.0", "Pass min_db=None to do the same as librosa.")},
)

# Core: constant-Q ----------------------------------------------------------

_CQT_SHAPE = "(channels, n_bins, n_frames), also for mono audio; index [0] for librosa's mono shape."
_CQT_OPTIONS: dict[str, Option] = {**_AUDIO, "dtype": _DTYPE}
_DECIMATION = "Octave decimation uses this library's resampler, not soxr: the top octave matches to round-off, lower octaves to ~1e-4 of the peak."
_CQT_PARAMS = "y | sr hop_length fmin n_bins bins_per_octave tuning filter_scale norm sparsity window scale pad_mode res_type dtype"
_e("librosa.cqt", _CQT_PARAMS, "AudioSamples.cqt", WITH_DIFFERENCES, _DECIMATION, shape=_CQT_SHAPE, options=_CQT_OPTIONS)
_e(
    "librosa.vqt",
    "y | sr hop_length fmin n_bins intervals gamma bins_per_octave tuning filter_scale norm sparsity window scale pad_mode res_type dtype",
    "AudioSamples.vqt",
    WITH_DIFFERENCES,
    'As cqt.',
    shape=_CQT_SHAPE,
    options={**_CQT_OPTIONS},
)
_e(
    "librosa.pseudo_cqt",
    "y | sr hop_length fmin n_bins bins_per_octave tuning filter_scale norm sparsity window scale pad_mode dtype",
    "AudioSamples.pseudo_cqt",
    EQUIVALENT,
    "No resampling involved.",
    shape=_CQT_SHAPE,
    options=_CQT_OPTIONS,
)
_e("librosa.hybrid_cqt", _CQT_PARAMS, "AudioSamples.hybrid_cqt", WITH_DIFFERENCES, "Low octaves as cqt.", shape=_CQT_SHAPE, options=_CQT_OPTIONS)
_CQT_SAMPLE_RATE = Default(22050, "a required sample_rate", "Pass sample_rate=...", value_note="becomes sample_rate=.")
_e(
    "librosa.icqt",
    "C | sr hop_length fmin bins_per_octave tuning filter_scale norm sparsity window scale length res_type dtype",
    "AudioSamples.icqt",
    WITH_DIFFERENCES,
    "A static method; upsampling uses this library's resampler.",
    options={"sr": _CQT_SAMPLE_RATE},
    behaviour=("Returns an AudioSamples.",),
)
_e(
    "librosa.griffinlim_cqt",
    "C | n_iter sr hop_length fmin bins_per_octave tuning filter_scale norm sparsity window scale pad_mode res_type dtype length momentum init rng random_state",
    "AudioSamples.griffinlim_cqt",
    WITH_DIFFERENCES,
    "A static method; init=None is the default; random phases come from this library's RNG.",
    options={
        "sr": _CQT_SAMPLE_RATE,
        "init": Default("random", "init=None", "librosa starts from random phases here; pass init='random' to do the same."),
        "rng": Default(None, "rng=0", "Repeated calls give the same output; pass rng=None for a new seed each call."),
    },
    behaviour=("Returns an AudioSamples.",),
)

# Core: pitch, tuning and harmonics -----------------------------------------

_e(
    "librosa.yin",
    "y | fmin fmax sr frame_length hop_length trough_threshold center pad_mode",
    ("AudioSamples.yin", "features.yin"),
    EQUIVALENT,
    "fmin and fmax are required, as in librosa.",
    options=_AUDIO,
    shape=_TRACK,
)
_e(
    "librosa.pyin",
    "y | fmin fmax sr frame_length hop_length n_thresholds beta_parameters boltzmann_parameter resolution max_transition_rate switch_prob no_trough_prob fill_na center pad_mode transition_min_prob",
    ("AudioSamples.pyin", "features.pyin"),
    EQUIVALENT,
    "Returns (f0, voiced_flag, voiced_prob).",
    options=_AUDIO,
    shape=_TRACK,
)
_e(
    "librosa.piptrack",
    "| y sr S n_fft hop_length fmin fmax threshold win_length window center pad_mode ref",
    ("AudioSamples.piptrack", "features.piptrack_from_spectrogram"),
    EQUIVALENT,
    options={**_AUDIO, "S": _NOT_METHOD},
    shape=_FRAMES,
    forms=[
        Form(
            ("S",),
            ("features.piptrack_from_spectrogram",),
            options={
                "y": _NOT_AUDIO,
                **_free("sr"),
                **_computed_elsewhere("hop_length", "win_length", "window", "center", "pad_mode"),
                "n_fft": _INFER_N_FFT,
            },
        )
    ],
)
_e(
    "librosa.estimate_tuning",
    "| y sr S n_fft resolution bins_per_octave **",
    ("AudioSamples.estimate_tuning", "features.estimate_tuning_from_signal", "features.estimate_tuning_from_spectrogram"),
    EQUIVALENT,
    "Pools all channels, as librosa does.",
    options={**_AUDIO, "S": _NOT_METHOD},
    forms=[
        Form(
            ("S",),
            ("features.estimate_tuning_from_spectrogram",),
            options={"y": _NOT_AUDIO, **_free("sr"), "n_fft": _INFER_N_FFT},
        )
    ],
)
_e("librosa.pitch_tuning", "frequencies | resolution bins_per_octave", "features.pitch_tuning", EQUIVALENT)
_e("librosa.salience", "S | freqs harmonics weights aggregate filter_peaks fill_value kind axis", "features.salience", WITH_DIFFERENCES, "Linear interpolation along the frequency axis; forms.salience takes the others.", options={"kind": Only(("linear",), "Use forms.salience(S, ..., kind=...)."), "axis": Only((-2,), "Use forms.salience(S, ..., axis=...).")})
_e(
    "librosa.interp_harmonics",
    "x | freqs harmonics kind fill_value axis",
    "features.interp_harmonics",
    EQUIVALENT,
    "Interpolants other than linear, and other axes, need a 1-D frequency grid.",
)
_e(
    "librosa.f0_harmonics",
    "x | f0 freqs harmonics kind fill_value axis",
    "features.f0_harmonics",
    EQUIVALENT,
    "Interpolants other than linear, and other axes, need a 1-D frequency grid.",
)

# Core: unit conversions ----------------------------------------------------

_SR_KW = Default(22050, "a required sr", "Pass sr=... or sample_rate=...")
_POS_SR = Default(22050, "a required second argument", "Pass the sample rate positionally.", value_note="becomes the positional sample_rate.")
_e("librosa.frames_to_samples", "frames | hop_length n_fft", "utils.frames_to_samples", EQUIVALENT, options=_free("hop_length"))
_e(
    "librosa.frames_to_time",
    "frames | sr hop_length n_fft",
    "utils.frames_to_time",
    WITH_DIFFERENCES,
    "Positional (frames, sample_rate, hop_size); n_fft offset and negative frames supported.",
    options={
        "sr": _POS_SR,
        "hop_length": Default(512, "a required third argument", "Pass the hop positionally.", value_note="becomes the positional hop_size."),
    },
)
_e("librosa.samples_to_frames", "samples | hop_length n_fft", "utils.samples_to_frames", EQUIVALENT, options=_free("hop_length"))
_e(
    "librosa.samples_to_time",
    "samples | sr",
    "utils.samples_to_time",
    WITH_DIFFERENCES,
    "Positional (samples, sample_rate); scalar or array of non-negative indices.",
    options={"sr": _POS_SR},
)
_e(
    "librosa.time_to_frames",
    "times | sr hop_length n_fft",
    "utils.time_to_frames_floor",
    EQUIVALENT,
    "utils.time_to_frames keeps its older round-to-nearest behaviour.",
    options={"sr": _SR_KW, "hop_length": _ARRAY_HOP},
)
_e(
    "librosa.time_to_samples",
    "times | sr",
    "utils.time_to_samples",
    EQUIVALENT,
    "utils.seconds_to_samples rounds instead of truncating.",
    options={"sr": _SR_KW},
)
_e("librosa.blocks_to_frames", "blocks | block_length", "utils.blocks_to_frames", EQUIVALENT)
_e("librosa.blocks_to_samples", "blocks | block_length hop_length", "utils.blocks_to_samples", EQUIVALENT)
_e(
    "librosa.blocks_to_time",
    "blocks | block_length hop_length sr",
    "utils.blocks_to_time",
    EQUIVALENT,
    options={"sr": Renamed("sample_rate")},
)
_e("librosa.times_like", "X | sr hop_length n_fft axis", "utils.times_like", EQUIVALENT, options={"sr": _SR_KW, "hop_length": _ARRAY_HOP})
_e("librosa.samples_like", "X | hop_length n_fft axis", "utils.samples_like", EQUIVALENT, options=_free("hop_length"))
_e(
    "librosa.fft_frequencies",
    "| sr n_fft",
    "utils.fft_frequencies",
    WITH_DIFFERENCES,
    "Positional (n_fft, sample_rate) with no defaults; same values.",
    options={
        "sr": _POS_SR,
        "n_fft": Default(2048, "a required first argument", "Pass the FFT size positionally.", value_note="becomes positional."),
    },
)
_HTK = Default(False, "htk=True", "The default scale here is HTK; pass htk=False for the Slaney scale.")
_e(
    "librosa.mel_frequencies",
    "n_mels | fmin fmax htk",
    "utils.mel_frequencies",
    WITH_DIFFERENCES,
    "With htk=False. The default here is HTK, kept for compatibility.",
    options={
        "n_mels": Default(128, "a required n_mels", "Pass n_mels=128."),
        "fmin": Default(0.0, "a required fmin", "Pass fmin=0.0."),
        "fmax": Default(11025.0, "a required fmax", "Pass fmax=11025.0."),
        "htk": _HTK,
    },
)
_e("librosa.hz_to_mel", "frequencies | htk", "utils.hz_to_mel", WITH_DIFFERENCES, "With htk=False.", options={"frequencies": Renamed("freq_hz"), "htk": _HTK})
_e("librosa.mel_to_hz", "mels | htk", "utils.mel_to_hz", WITH_DIFFERENCES, "With htk=False.", options={"mels": Renamed("mel"), "htk": _HTK})
_e("librosa.cqt_frequencies", "n_bins | fmin bins_per_octave tuning", "utils.cqt_frequencies", EQUIVALENT)
_e("librosa.tempo_frequencies", "n_bins | hop_length sr", "rhythm.tempo_frequencies", EQUIVALENT, options=_free("sr"))
_e("librosa.fourier_tempo_frequencies", "| sr win_length hop_length", "rhythm.fourier_tempo_frequencies", EQUIVALENT, options=_free("sr"))
_e("librosa.hz_to_midi", "frequencies", "utils.hz_to_midi", EQUIVALENT, options={"frequencies": Renamed("freq_hz")})
_e("librosa.midi_to_hz", "notes", "utils.midi_to_hz", EQUIVALENT, options={"notes": Renamed("midi_note")})
_e(
    "librosa.hz_to_note",
    "frequencies | **",
    "utils.frequency_to_note",
    WITH_DIFFERENCES,
    "Returns (name, cents) (a list of names and a cents array for array input); ASCII #; MIDI 0..127 only.",
    options={
        "frequencies": Renamed("freq_hz"),
        "octave": Unsupported("use notation.hz_to_note(hz, octave=...)."),
        "cents": Absorbed("the cents are always returned."),
        "key": Unsupported("use notation.hz_to_note(hz, key=...)."),
        "unicode": Only((False,), "Use notation.hz_to_note(hz, unicode=True)."),
    },
)
_e(
    "librosa.midi_to_note",
    "midi | octave cents key unicode",
    "utils.midi_to_note",
    WITH_DIFFERENCES,
    "Integer MIDI 0..127 (an array gives a list of names); ASCII #; no cents, no octave=False.",
    options={
        "midi": Renamed("midi_note"),
        "octave": Only((True,), "Use notation.midi_to_note(midi, octave=False)."),
        "cents": Only((False,), "Use notation.midi_to_note(midi, cents=True)."),
        "key": Only(("C:maj",), "Use notation.midi_to_note(midi, key=...)."),
        "unicode": Default(True, "ASCII #", "librosa returns Unicode sharps here; compare names with that in mind."),
    },
)
_e(
    "librosa.note_to_hz",
    "note | round_midi",
    "utils.note_to_frequency",
    WITH_DIFFERENCES,
    "Name or sequence of names; grammar and limits as note_to_midi.",
    options={"note": Renamed("note_name"), "round_midi": Only((False,), "Use notation.note_to_hz(note, round_midi=True).")},
)
_e(
    "librosa.note_to_midi",
    "note | round_midi",
    "utils.note_to_midi",
    WITH_DIFFERENCES,
    "Name or sequence of names; librosa's grammar (accidentals, cents suffix); the octave is required and the result is in 0..127.",
    options={"note": Renamed("note_name"), "round_midi": Only((True,), "Use notation.note_to_midi(note, round_midi=False).")},
)
_e("librosa.hz_to_octs", "frequencies | tuning bins_per_octave", "utils.hz_to_octs", EQUIVALENT)
_e("librosa.octs_to_hz", "octs | tuning bins_per_octave", "utils.octs_to_hz", EQUIVALENT)
_e("librosa.A4_to_tuning", "A4 | bins_per_octave", "utils.a4_to_tuning", EQUIVALENT, "Lower-case name.")
_e("librosa.tuning_to_A4", "tuning | bins_per_octave", "utils.tuning_to_a4", EQUIVALENT, "Lower-case name.")
_e("librosa.key_to_notes", "key | unicode natural", "utils.key_to_notes", EQUIVALENT)
_e("librosa.key_to_degrees", "key", "utils.key_to_degrees", EQUIVALENT)
_e(
    "librosa.interval_frequencies",
    "n_bins | fmin intervals bins_per_octave tuning sort",
    "notation.interval_frequencies",
    EQUIVALENT,
)
_e("librosa.pythagorean_intervals", "| bins_per_octave sort return_factors", "notation.pythagorean_intervals", EQUIVALENT)
_e("librosa.plimit_intervals", "| primes bins_per_octave sort return_factors", "notation.plimit_intervals", EQUIVALENT)
_e(
    "librosa.interval_to_fjs",
    "interval | unison tolerance unicode",
    "notation.interval_to_fjs",
    WITH_DIFFERENCES,
    "Recognises more ratios: any built from the primes up to 23, where librosa looks up a fixed table.",
)
_e("librosa.hz_to_fjs", "frequencies | fmin unison unicode", "notation.hz_to_fjs", WITH_DIFFERENCES, "Recognises more ratios, as interval_to_fjs.")
_e("librosa.fifths_to_note", "| unison fifths unicode", "notation.fifths_to_note", EQUIVALENT)
_e("librosa.hz_to_svara_c", "frequencies | Sa mela abbr octave unicode", "notation.hz_to_svara_c", EQUIVALENT)
_e("librosa.hz_to_svara_h", "frequencies | Sa abbr octave unicode", "notation.hz_to_svara_h", EQUIVALENT)
_e("librosa.midi_to_svara_c", "midi | Sa mela abbr octave unicode", "notation.midi_to_svara_c", EQUIVALENT)
_e("librosa.midi_to_svara_h", "midi | Sa abbr octave unicode", "notation.midi_to_svara_h", EQUIVALENT)
_e("librosa.note_to_svara_c", "notes | Sa mela abbr octave unicode", "notation.note_to_svara_c", EQUIVALENT)
_e("librosa.note_to_svara_h", "notes | Sa abbr octave unicode", "notation.note_to_svara_h", EQUIVALENT)
_e("librosa.list_mela", "", "notation.list_mela", EQUIVALENT)
_e("librosa.list_thaat", "", "notation.list_thaat", EQUIVALENT)
_e("librosa.mela_to_degrees", "mela", "notation.mela_to_degrees", EQUIVALENT)
_e("librosa.mela_to_svara", "mela | abbr unicode", "notation.mela_to_svara", EQUIVALENT)
_e("librosa.thaat_to_degrees", "thaat", "notation.thaat_to_degrees", EQUIVALENT)

# librosa.feature -----------------------------------------------------------

_e(
    "librosa.feature.melspectrogram",
    f"| y sr S {_STFT_PARAMS} power **",
    ("AudioSamples.melspectrogram", "spectral.melspectrogram_from_spectrogram"),
    WITH_DIFFERENCES,
    "float64 mel basis (librosa's default float32 basis: ~1e-7 relative).",
    options={**_AUDIO, "S": _NOT_METHOD, "dtype": _MEL_DTYPE},
    shape=_FRAMES,
    forms=[
        Form(
            ("S",),
            ("spectral.melspectrogram_from_spectrogram",),
            options={
                "y": _NOT_AUDIO,
                **_free("sr"),
                **_computed_elsewhere("hop_length", "win_length", "window", "center", "pad_mode", "power"),
                "n_fft": _INFER_N_FFT,
                "dtype": _MEL_DTYPE,
            },
        )
    ],
)
_e(
    "librosa.feature.mfcc",
    "| y sr S n_mfcc dct_type norm lifter mel_norm **",
    ("AudioSamples.mfcc_frames", "spectral.mfcc_from_mel_db"),
    WITH_DIFFERENCES,
    "float64 mel basis; n_mfcc > n_mels raises.",
    options={**_AUDIO, "S": _NOT_METHOD, "dtype": _MEL_DTYPE},
    shape=_FRAMES,
    behaviour=("librosa always applies an 80 dB floor to the log mel spectrogram; mfcc_frames does so only with top_db=80.0.",),
    forms=[
        Form(
            ("S",),
            ("spectral.mfcc_from_mel_db",),
            options={"y": _NOT_AUDIO, "sr": Absorbed("not used for a mel spectrogram in dB."), "mel_norm": Absorbed("only used to compute a mel spectrogram from audio.")},
            shape="As the input, with n_mfcc rows.",
            behaviour=("S is a log mel spectrogram in dB, as in librosa.",),
        )
    ],
)
_e("librosa.feature.delta", "data | width order axis mode **", "spectral.delta", EQUIVALENT)
_e("librosa.feature.stack_memory", "data | n_steps delay **", "spectral.stack_memory", EQUIVALENT)


def _spectral_feature(
    short: str,
    params: str,
    match: str,
    note: str = "",
    *,
    method_extra: Mapping[str, Option] | None = None,
) -> None:
    """A per-frame scalar with an audio method and a from-spectrogram function."""
    from_spectrogram = f"features.{short}_from_spectrogram"
    uses_sr = " sr " in params
    options: dict[str, Option] = {"y": _Y, "S": _NOT_METHOD, **dict(method_extra or {})}
    if " freq" in params:
        options["freq"] = Unsupported(f"the method uses the bin frequencies; use {from_spectrogram}(S, freq=...).")
    form: dict[str, Option] = {"y": _NOT_AUDIO, **_computed_elsewhere(*_STFT_KWS)}
    if uses_sr:
        options["sr"] = _SR
        form["sr"] = _ARRAY_SR
    _e(
        f"librosa.feature.{short}",
        params,
        (f"AudioSamples.{short}_frames", from_spectrogram),
        match,
        note,
        options=options,
        shape=_SCALARS,
        forms=[Form(("S",), (from_spectrogram,), options=form, shape="librosa's shape: (channels..., 1, n_frames).")],
    )


_spectral_feature("spectral_centroid", f"| y sr S {_STFT_PARAMS} freq", EQUIVALENT)
_spectral_feature(
    "spectral_bandwidth",
    f"| y sr S {_STFT_PARAMS} freq centroid norm p",
    EQUIVALENT,
    method_extra={"centroid": Unsupported("use features.spectral_bandwidth_from_spectrogram(S, centroid=...).")},
)
_spectral_feature("spectral_contrast", f"| y sr S {_STFT_PARAMS} freq fmin n_bands quantile linear", WITH_DIFFERENCES, "A band with no FFT bin raises.")
_spectral_feature("spectral_flatness", f"| y S {_STFT_PARAMS} amin power", EQUIVALENT)
_spectral_feature("spectral_rolloff", f"| y sr S {_STFT_PARAMS} freq roll_percent", EQUIVALENT)
_e(
    "librosa.feature.poly_features",
    f"| y sr S {_STFT_PARAMS} order freq",
    ("AudioSamples.poly_features", "features.poly_features_from_spectrogram"),
    WITH_DIFFERENCES,
    "A rank-deficient fit raises (NumPy warns and returns a minimum-norm solution).",
    options={
        **_AUDIO,
        "S": _NOT_METHOD,
        "freq": Unsupported("the method uses the bin frequencies; use features.poly_features_from_spectrogram(S, freq=...)."),
    },
    shape=_FRAMES,
    forms=[
        Form(
            ("S",),
            ("features.poly_features_from_spectrogram",),
            options={"y": _NOT_AUDIO, **_computed_elsewhere(*_STFT_KWS), "sr": _ARRAY_SR},
        )
    ],
)
_e(
    "librosa.feature.rms",
    "| y S frame_length hop_length center pad_mode dtype",
    ("AudioSamples.rms_frames", "features.rms_from_spectrogram"),
    WITH_DIFFERENCES,
    "Output precision follows the sample type (librosa's default dtype=float32).",
    options={"y": _Y, "S": _NOT_METHOD, "dtype": _DTYPE},
    shape=_SCALARS,
    forms=[
        Form(
            ("S",),
            ("features.rms_from_spectrogram",),
            options={
                "y": _NOT_AUDIO,
                **_computed_elsewhere("hop_length", "center", "pad_mode"),
                "dtype": Unsupported("the output precision follows the array."),
            },
        )
    ],
)
_e(
    "librosa.feature.zero_crossing_rate",
    "y | frame_length hop_length center **",
    "AudioSamples.zero_crossing_rate_frames",
    EQUIVALENT,
    options={"y": _Y},
    shape=_SCALARS,
)
_TUNING = Default(None, "tuning=0.0", "librosa estimates the tuning here; pass tuning=None to do the same.")
_e(
    "librosa.feature.chroma_stft",
    "| y sr S norm n_fft hop_length win_length window center pad_mode tuning n_chroma **",
    ("AudioSamples.chroma_stft", "features.chroma_stft_from_spectrogram"),
    WITH_DIFFERENCES,
    "float64 chroma basis; with odd n_fft the given n_fft is used for tuning.",
    options={**_AUDIO, "S": _NOT_METHOD, "tuning": _TUNING},
    shape=_FRAMES,
    forms=[
        Form(
            ("S",),
            ("features.chroma_stft_from_spectrogram",),
            options={
                "y": _NOT_AUDIO,
                **_computed_elsewhere("hop_length", "win_length", "window", "center", "pad_mode"),
                "n_fft": _INFER_N_FFT,
                "sr": _ARRAY_SR,
                "tuning": _TUNING,
            },
        )
    ],
)
_CHROMA_WINDOW = Only((None,), "Compute the CQT, then use forms.chroma_cqt(C=..., window=...) or forms.chroma_cens(C=..., window=...).")
_C_FORM = Unsupported("use forms.chroma_cqt(C=...) or forms.chroma_cens(C=...).")
_e(
    "librosa.feature.chroma_cqt",
    "| y sr C hop_length fmin norm threshold tuning n_chroma n_octaves window bins_per_octave cqt_mode",
    "AudioSamples.chroma_cqt",
    WITH_DIFFERENCES,
    "Inherits the CQT resampler difference; window must be None.",
    options={**_AUDIO, "C": _C_FORM, "tuning": _TUNING, "window": _CHROMA_WINDOW},
    shape=_FRAMES,
)
_e(
    "librosa.feature.chroma_cens",
    "| y sr C hop_length fmin tuning n_chroma n_octaves bins_per_octave cqt_mode window norm win_len_smooth smoothing_window",
    ("AudioSamples.chroma_cens", "features.chroma_cens_from_chroma"),
    WITH_DIFFERENCES,
    "As chroma_cqt from audio; equivalent from a chromagram with features.chroma_cens_from_chroma.",
    options={**_AUDIO, "C": _C_FORM, "tuning": _TUNING, "window": _CHROMA_WINDOW},
    shape=_FRAMES,
)
_e(
    "librosa.feature.chroma_vqt",
    "| y sr V hop_length fmin intervals norm threshold n_octaves bins_per_octave gamma",
    "AudioSamples.chroma_vqt",
    WITH_DIFFERENCES,
    "As vqt.",
    options={
        **_AUDIO,
        "V": Unsupported("use forms.chroma_vqt(V=...)."),
        "intervals": Only(("equal",), "Use AudioSamples.vqt(intervals=...), then forms.chroma_vqt(V=...)."),
    },
    shape=_FRAMES,
)
_e(
    "librosa.feature.tonnetz",
    "| y sr chroma **",
    ("AudioSamples.tonnetz", "features.tonnetz_from_chroma"),
    WITH_DIFFERENCES,
    "As chroma_cqt from audio; equivalent from a chromagram with features.tonnetz_from_chroma.",
    options={**_AUDIO, "chroma": _NOT_METHOD},
    shape=_FRAMES,
    behaviour=("tuning defaults to 0.0 here (librosa estimates it through chroma_cqt); pass tuning=None to do the same.",),
    forms=[
        Form(
            ("chroma",),
            ("features.tonnetz_from_chroma",),
            options={"y": Absorbed("this form takes a chromagram rather than audio."), "sr": Absorbed("a chromagram does not need it.")},
        )
    ],
)
_ENVELOPE = "onset_envelope"
_ENV_OPTION: dict[str, Option] = {**_AUDIO, _ENVELOPE: _NOT_METHOD}


def _envelope_form(function: str) -> Form:
    return Form(
        (_ENVELOPE,),
        (function,),
        options={"y": Absorbed("this form takes an onset envelope rather than audio."), "sr": _ARRAY_SR},
    )


_e(
    "librosa.feature.tempogram",
    "| y sr onset_envelope hop_length win_length center window norm",
    ("AudioSamples.tempogram", "rhythm.tempogram"),
    EQUIVALENT,
    options=_ENV_OPTION,
    shape=_FRAMES,
    forms=[_envelope_form("rhythm.tempogram")],
)
_e(
    "librosa.feature.fourier_tempogram",
    "| y sr onset_envelope hop_length win_length center window",
    ("AudioSamples.fourier_tempogram", "rhythm.fourier_tempogram"),
    EQUIVALENT,
    options=_ENV_OPTION,
    shape=_FRAMES,
    forms=[_envelope_form("rhythm.fourier_tempogram")],
)
_e(
    "librosa.feature.tempo",
    "| y sr onset_envelope tg hop_length start_bpm std_bpm ac_size max_tempo aggregate prior",
    ("AudioSamples.tempo", "rhythm.tempo"),
    EQUIVALENT,
    options={**_ENV_OPTION, "tg": _NOT_METHOD},
    shape="(channels, 1), or (channels, n_frames) when time-varying; index [0] for librosa's mono shape.",
    forms=[
        Form(
            (_ENVELOPE, "tg"),
            ("rhythm.tempo",),
            options={"y": Absorbed("this form takes an onset envelope or tempogram rather than audio."), "sr": _ARRAY_SR},
        )
    ],
)
_e(
    "librosa.feature.tempogram_ratio",
    "| y sr onset_envelope tg bpm hop_length win_length start_bpm std_bpm max_tempo freqs factors aggregate prior center window kind fill_value norm",
    ("AudioSamples.tempogram_ratio", "rhythm.tempogram_ratio"),
    WITH_DIFFERENCES,
    "Custom freqs not supported.",
    options={
        **_ENV_OPTION,
        "tg": Unsupported("use rhythm.tempogram_ratio(tg=...)."),
        "bpm": Unsupported("use rhythm.tempogram_ratio(bpm=...)."),
        "freqs": Unsupported("use forms.tempogram_ratio(tg=..., bpm=..., freqs=...)."),
        "kind": Unsupported("use rhythm.tempogram_ratio(kind=...)."),
    },
    shape=_FRAMES,
    forms=[
        Form(
            (_ENVELOPE, "tg"),
            ("rhythm.tempogram_ratio",),
            options={
                "y": Absorbed("this form takes an onset envelope or tempogram rather than audio."),
                "sr": _ARRAY_SR,
                "freqs": Unsupported("use forms.tempogram_ratio(tg=..., bpm=..., freqs=...)."),
            },
        )
    ],
)
_e(
    "librosa.feature.hybrid_tempogram",
    "| y sr onset_envelope hop_length win_length center window **",
    ("AudioSamples.hybrid_tempogram", "rhythm.hybrid_tempogram"),
    EQUIVALENT,
    options=_ENV_OPTION,
    shape=_FRAMES,
    forms=[_envelope_form("rhythm.hybrid_tempogram")],
)
_e(
    "librosa.feature.metrogram",
    "| tg freqs factors aggregate kind fill_value",
    "rhythm.metrogram",
    WITH_DIFFERENCES,
    "The aggregate runs in the tempogram's float type.",
)
_e(
    "librosa.feature.inverse.mel_to_stft",
    "M | sr n_fft power **",
    "inverse.mel_to_stft",
    WITH_DIFFERENCES,
    "Coordinate descent reaches a better fit than librosa's early-stopped L-BFGS-B; the magnitudes differ by up to 15 % in energy.",
    options={"sr": _ARRAY_SR},
)
_e(
    "librosa.feature.inverse.mel_to_audio",
    "M | sr n_fft hop_length win_length window center pad_mode power n_iter length dtype **",
    "inverse.mel_to_audio",
    WITH_DIFFERENCES,
    "As mel_to_stft, then fast Griffin-Lim; random phases come from this library's generator, seeded by rng.",
    options={"sr": _ARRAY_SR, "dtype": _DTYPE},
)
_e(
    "librosa.feature.inverse.mfcc_to_mel",
    "mfcc | n_mels dct_type norm ref lifter",
    "inverse.mfcc_to_mel",
    EQUIVALENT,
)
_e(
    "librosa.feature.inverse.mfcc_to_audio",
    "mfcc | n_mels dct_type norm ref lifter **",
    "inverse.mfcc_to_audio",
    WITH_DIFFERENCES,
    "sr is required; the filterbank's norm is mel_norm, because norm is the DCT's. As mel_to_audio otherwise.",
    options={"sr": _ARRAY_SR, "dtype": _DTYPE},
)

# librosa.effects -----------------------------------------------------------

_e(
    "librosa.effects.time_stretch",
    "y | rate **",
    "AudioSamples.time_stretch",
    EQUIVALENT,
    "Exactly round(n / rate) samples.",
    options={"y": _Y},
    behaviour=("Returns an AudioSamples of the input's sample type.",),
)
_e(
    "librosa.effects.pitch_shift",
    "y | sr n_steps bins_per_octave res_type scale **",
    "AudioSamples.pitch_shift",
    WITH_DIFFERENCES,
    "Final resampling by this library's resampler, not soxr (~68 dB SNR).",
    options={**_AUDIO, "scale": Only((False,), "Use forms.pitch_shift(y, ..., scale=True).")},
    behaviour=("Returns an AudioSamples of the input's sample type.",),
)
_ZI: dict[str, Option] = {
    "y": _Y,
    "zi": Unsupported("use effects.preemphasis or effects.deemphasis on an array for filter state."),
    "return_zf": Unsupported("use effects.preemphasis or effects.deemphasis on an array for filter state."),
}
_e("librosa.effects.preemphasis", "y | coef zi return_zf", ("AudioSamples.preemphasis", "effects.preemphasis"), EQUIVALENT, "The function takes zi and return_zf.", options=_ZI)
_e("librosa.effects.deemphasis", "y | coef zi return_zf", ("AudioSamples.deemphasis", "effects.deemphasis"), EQUIVALENT, options=_ZI)
_TRIM_TOP_DB = Default(60, "a required top_db", "Pass top_db=60 to do the same.")
_MAX_OVER_CHANNELS = Same("None is the maximum over channels.")
_RMS_NOTE = "frame RMS in the sample type's float (librosa: float32)"
_CALLABLE_ONLY = "A callable ref or aggregate runs with the GIL held."
_e(
    "librosa.effects.trim",
    "y | top_db ref frame_length hop_length aggregate",
    "AudioSamples.trim_db",
    WITH_DIFFERENCES,
    f"Returns (trimmed, array([start, end])); raises when everything is silent (librosa returns an empty array); {_RMS_NOTE}. {_CALLABLE_ONLY}",
    options={"y": _Y, "top_db": _TRIM_TOP_DB, "aggregate": _MAX_OVER_CHANNELS},
)
_e(
    "librosa.effects.split",
    "y | top_db ref frame_length hop_length aggregate",
    "AudioSamples.split_db",
    WITH_DIFFERENCES,
    f"int64 (n, 2) intervals; {_RMS_NOTE}. {_CALLABLE_ONLY}",
    options={"y": _Y, "top_db": _TRIM_TOP_DB, "aggregate": _MAX_OVER_CHANNELS},
)
_e(
    "librosa.effects.remix",
    "y intervals | align_zeros",
    "AudioSamples.remix",
    EQUIVALENT,
    options={
        "y": _Y,
        "align_zeros": Default(True, "align_zeros=False", "librosa snaps boundaries to zero crossings here; pass align_zeros=True to do the same."),
    },
)
_HPSS_PARAMS = f"y | kernel_size power mask margin {_STFT_PARAMS}"
_e(
    "librosa.effects.hpss",
    _HPSS_PARAMS,
    "AudioSamples.harmonic_percussive",
    WITH_DIFFERENCES,
    "window is used (librosa ignores it). AudioSamples.hpss is a different method.",
    options={"y": _Y},
    behaviour=("Returns two AudioSamples of the input's sample type.",),
)
_MASK_FALSE = Only((False,), "librosa returns the inverse STFT of the mask, not the masks; for the masks use effects.hpss(S, mask=True) on a spectrogram.")
_e("librosa.effects.harmonic", _HPSS_PARAMS, "AudioSamples.harmonic", WITH_DIFFERENCES, "As hpss; mask=True is not supported.", options={"y": _Y, "mask": _MASK_FALSE})
_e("librosa.effects.percussive", _HPSS_PARAMS, "AudioSamples.percussive", WITH_DIFFERENCES, "As harmonic.", options={"y": _Y, "mask": _MASK_FALSE})

# librosa.onset and librosa.beat --------------------------------------------

_e(
    "librosa.onset.onset_strength",
    "| y sr S lag max_size ref detrend center feature aggregate **",
    ("AudioSamples.onset_strength", "rhythm.onset_strength"),
    EQUIVALENT,
    "float64 mel basis. Keeps librosa's 80 dB floor.",
    options={
        **_AUDIO,
        "S": _NOT_METHOD,
        "ref": Unsupported("use rhythm.onset_strength(S=..., ref=...)."),
        "feature": Unsupported("use forms.onset_strength(y=..., feature=...)."),
    },
    shape=_TRACK,
    forms=[
        Form(
            ("S",),
            ("rhythm.onset_strength",),
            options={"y": _NOT_AUDIO, "sr": _ARRAY_SR, "feature": Unsupported("the features are given as S.")},
        )
    ],
)
_e(
    "librosa.onset.onset_strength_multi",
    "| y sr S n_fft hop_length lag max_size ref detrend center feature aggregate channels **",
    ("AudioSamples.onset_strength_multi", "rhythm.onset_strength_multi"),
    WITH_DIFFERENCES,
    "The method's channels count mel bands.",
    options={
        **_AUDIO,
        "S": _NOT_METHOD,
        "ref": Unsupported("use rhythm.onset_strength_multi(S=..., ref=...)."),
        "feature": Unsupported("the feature is always the log mel spectrogram; compute your own and pass S= to rhythm.onset_strength_multi."),
    },
    forms=[
        Form(
            ("S",),
            ("rhythm.onset_strength_multi",),
            options={"y": _NOT_AUDIO, "sr": _ARRAY_SR, "feature": Unsupported("the features are given as S.")},
        )
    ],
)
_SPARSE_ONLY = Only((True,), "Use forms.onset_detect(y=..., sparse=False) or forms.beat_track(y=..., sparse=False).")
_e(
    "librosa.onset.onset_detect",
    "| y sr onset_envelope hop_length backtrack energy units normalize sparse **",
    ("AudioSamples.onset_detect", "rhythm.onset_detect"),
    WITH_DIFFERENCES,
    "Mono audio only; sparse=True only.",
    options={**_ENV_OPTION, "sparse": _SPARSE_ONLY},
    forms=[
        Form(
            (_ENVELOPE,),
            ("rhythm.onset_detect",),
            options={"y": Absorbed("this form takes an onset envelope rather than audio."), "sr": _ARRAY_SR, "sparse": _SPARSE_ONLY},
        )
    ],
)
_e("librosa.onset.onset_backtrack", "events energy", "rhythm.onset_backtrack", WITH_DIFFERENCES, "No events give an empty array (librosa raises).")
_BEAT_TRIM = Default(True, "trim=False", "librosa drops weak leading and trailing beats here; pass trim=True to do the same.")
_e(
    "librosa.beat.beat_track",
    "| y sr onset_envelope hop_length start_bpm tightness trim bpm prior units sparse",
    ("AudioSamples.beat_track", "rhythm.beat_track", "rhythm.beat_track_with_bpm"),
    WITH_DIFFERENCES,
    "Mono audio only; sparse=True only.",
    options={**_ENV_OPTION, "trim": _BEAT_TRIM, "sparse": _SPARSE_ONLY},
    forms=[
        Form(
            (_ENVELOPE,),
            ("rhythm.beat_track",),
            options={"y": Absorbed("this form takes an onset envelope rather than audio."), "sr": _ARRAY_SR, "trim": _BEAT_TRIM, "sparse": _SPARSE_ONLY},
        )
    ],
)
_e(
    "librosa.beat.plp",
    "| y sr onset_envelope hop_length win_length tempo_min tempo_max prior",
    ("AudioSamples.plp", "rhythm.plp"),
    EQUIVALENT,
    options=_ENV_OPTION,
    shape=_TRACK,
    forms=[_envelope_form("rhythm.plp")],
)

# librosa.decompose ---------------------------------------------------------

_e(
    "librosa.decompose.decompose",
    "S | n_components transformer sort fit **",
    "factorization.decompose",
    WITH_DIFFERENCES,
    "No scikit-learn needed: the default transformer is factorization.Nmf, which takes NMF's arguments; any object with fit_transform, transform and components_ also works. Same solvers; a different random generator in the SVD start.",
    options={
        "random_state": Default(None, "random_state=0", "Repeated calls give the same result."),
        "verbose": Unsupported("the factorisation prints nothing."),
    },
)
_e("librosa.decompose.hpss", "S | kernel_size power mask margin", "effects.hpss", EQUIVALENT, "Kernel sizes, power, margins, mask=True.")
_e(
    "librosa.decompose.nn_filter",
    "S | rec aggregate axis **",
    "effects.nn_filter",
    WITH_DIFFERENCES,
    "Dense or scipy.sparse recurrence matrices; any aggregate (a callable runs with the GIL held).",
)

# librosa.filters -----------------------------------------------------------

_e("librosa.filters.get_window", "window Nx | fftbins", "spectral.get_window", EQUIVALENT, "Periodic (fftbins=True) or symmetric.")
_FILTER_DTYPE = Only((), "Tables are computed and returned in float64; convert the array if you need another dtype.", names=("np.float64", "numpy.float64"))
_WAVELET_DTYPE = Only((), "Filters are computed and returned in complex128; convert the array if you need another dtype.", names=("np.complex128", "numpy.complex128"))
_e(
    "librosa.filters.mel",
    "| sr n_fft n_mels fmin fmax htk norm dtype",
    "spectral.mel_filterbank",
    WITH_DIFFERENCES,
    "float64 weights (librosa dtype=float32); empty filters are zero rows without a warning.",
    options={"dtype": _FILTER_DTYPE},
)
_e(
    "librosa.filters.chroma",
    "| sr n_fft n_chroma tuning ctroct octwidth norm base_c dtype",
    "features.chroma_filterbank",
    WITH_DIFFERENCES,
    "float64 weights.",
    options={"dtype": _FILTER_DTYPE},
)
_e(
    "librosa.filters.cq_to_chroma",
    "n_input | bins_per_octave n_chroma fmin window base_c dtype",
    "features.cq_to_chroma",
    WITH_DIFFERENCES,
    "float64 tables.",
    options={"dtype": _FILTER_DTYPE},
)
_e(
    "librosa.filters.wavelet",
    "| freqs sr window filter_scale pad_fft norm dtype gamma alpha **",
    "features.wavelet",
    WITH_DIFFERENCES,
    "complex128 filters (librosa: complex64).",
    options={"sr": _ARRAY_SR, "dtype": _WAVELET_DTYPE},
)
_e(
    "librosa.filters.wavelet_lengths",
    "| freqs sr window filter_scale gamma alpha",
    "features.wavelet_lengths",
    WITH_DIFFERENCES,
    "An explicit alpha allows one frequency; without it, at least two ascending frequencies.",
    options={"sr": _ARRAY_SR},
)
_e(
    "librosa.filters.window_bandwidth",
    "window n",
    "spectral.window_bandwidth",
    WITH_DIFFERENCES,
    "Always evaluates the formula; librosa returns a table value for named windows (up to ~1e-4 relative apart).",
)
_e(
    "librosa.filters.window_sumsquare",
    "| window n_frames hop_length win_length n_fft dtype norm",
    "spectral.window_sumsquare",
    EQUIVALENT,
    "float64 output unless dtype=numpy.float32 (librosa's default).",
    options={"dtype": Default(_Named("numpy.float32"), "dtype=None (float64)", "Pass dtype=numpy.float32 to do the same.")},
)
_e("librosa.filters.diagonal_filter", "window n | slope angle zero_mean", "segment.diagonal_filter", EQUIVALENT)
_e("librosa.filters.mr_frequencies", "tuning", "pitch_filterbank.mr_frequencies", EQUIVALENT)
_e(
    "librosa.filters.semitone_filterbank",
    "| center_freqs tuning sample_rates flayout **",
    "pitch_filterbank.semitone_filterbank",
    EQUIVALENT,
    options={
        "flayout": Default("ba", "flayout='sos'", "Second-order sections keep the precision that polynomials lose at these orders. Pass flayout='ba' for polynomials."),
    },
)

# librosa.util --------------------------------------------------------------

_e(
    "librosa.util.frame",
    "x | frame_length hop_length axis writeable subok",
    "spectral.frame",
    WITH_DIFFERENCES,
    "1-D input, last-axis framing. A writable copy by default; copy=False gives librosa's read-only strided view.",
    options={
        "axis": Only((-1,), "Use forms.frame(x, ..., axis=...)."),
        "writeable": Renamed("copy", "copy=True gives a writable array; copy=False a read-only view."),
        "subok": Unsupported("use forms.frame(x, ..., subok=True)."),
    },
)
_e("librosa.util.normalize", "S | norm axis threshold fill", "features.normalize", EQUIVALENT)
_e("librosa.util.peak_pick", "x | pre_max post_max pre_avg post_avg delta wait sparse method axis", "rhythm.peak_pick", WITH_DIFFERENCES, "1-D input.")
_e("librosa.util.sync", "data idx | aggregate pad axis", "sequence.sync", WITH_DIFFERENCES, "An empty index list gives one segment.")
_e("librosa.util.softmask", "X X_ref | power split_zeros", "effects.softmask", EQUIVALENT)
_e(
    "librosa.util.fix_length",
    "data | size axis **",
    "AudioSamples.pad_samples_right",
    WITH_DIFFERENCES,
    "Pads only; a longer signal is returned unchanged.",
    options={"data": _Y, "size": Renamed("target_num_samples"), "axis": _LAYOUT},
)
_e(
    "librosa.util.nnls",
    "A B | **",
    "linalg.nnls",
    WITH_DIFFERENCES,
    "A matrix B is solved by coordinate descent from the same start, to a better objective; librosa stops L-BFGS-B early.",
    options={
        "bounds": Unsupported("the bound is always zero."),
    },
)
_none(
    "Argument-validation and deprecation plumbing.",
    "librosa.util.rename_kw(| old_name old_value new_name new_value version_deprecated version_removed)",
    "librosa.util.Deprecated()",
)
_INDEX64 = "Index outputs are int64."
_e("librosa.util.localmax", "x | axis", "arrays.localmax", EQUIVALENT)
_e("librosa.util.localmin", "x | axis", "arrays.localmin", EQUIVALENT)
_e("librosa.util.match_events", "events_from events_to left right", "arrays.match_events", WITH_DIFFERENCES, _INDEX64 + " Ties between targets at one position may pick another of the equal targets.")
_e(
    "librosa.util.match_intervals",
    "intervals_from intervals_to strict",
    "arrays.match_intervals",
    WITH_DIFFERENCES,
    _INDEX64 + " With strict=False an interval that overlaps nothing goes to the true nearest target; librosa skips the first target after it.",
)
_e("librosa.util.pad_center", "data | size axis **", "arrays.pad_center", EQUIVALENT)
_e("librosa.util.fix_frames", "frames | x_min x_max pad", "arrays.fix_frames", EQUIVALENT)
_e("librosa.util.abs2", "x dtype", "arrays.abs2", WITH_DIFFERENCES, "A complex scalar gives a float rather than a 0-d array.")
_e("librosa.util.phasor", "angles | mag", "arrays.phasor", WITH_DIFFERENCES, "Integer angles give complex128 rather than complex64: only float32 angles stay in single precision.")
_e("librosa.util.tiny", "x", "arrays.tiny", EQUIVALENT)
_e("librosa.util.expand_to", "x | ndim axes", "arrays.expand_to", WITH_DIFFERENCES, "Positions out of increasing order reorder the axes, so axis i is always axis axes[i]; librosa reinterprets the data.")
_e("librosa.util.stack", "arrays | axis", "arrays.stack", WITH_DIFFERENCES, "C order for every axis (librosa: Fortran order for axis=0). Same values.")
_e(
    "librosa.util.dtype_c2r",
    "d | default",
    "arrays.dtype_c2r",
    EQUIVALENT,
    options={"default": Same("None means float32.")},
)
_e(
    "librosa.util.dtype_r2c",
    "d | default",
    "arrays.dtype_r2c",
    EQUIVALENT,
    options={"default": Same("None means complex64.")},
)
_e(
    "librosa.util.MAX_MEM_BLOCK",
    "",
    "arrays.MAX_MEM_BLOCK",
    EQUIVALENT,
)
_e(
    "librosa.util.buf_to_float",
    "x | n_bytes dtype",
    "arrays.buf_to_float",
    WITH_DIFFERENCES,
    "Also decodes 3-byte samples. Only float32 and float64 outputs.",
    options={
        "dtype": Default(
            _Named("numpy.float32"),
            "float32 up to 3 bytes per sample, otherwise float64",
            "Pass dtype=numpy.float32 for 4- and 8-byte samples to do the same as librosa.",
        ),
    },
)
_e("librosa.util.valid_audio", "y", "arrays.valid_audio", EQUIVALENT)
_e(
    "librosa.util.axis_sort",
    "S | axis index value",
    "arrays.axis_sort",
    WITH_DIFFERENCES,
    "The sort is stable, so equal keys keep their order; librosa's order for equal keys is unspecified.",
)
_e(
    "librosa.util.sparsify_rows",
    "x | quantile dtype",
    "arrays.sparsify_rows",
    WITH_DIFFERENCES,
    "Returns arrays.SparseRows, with data, indices, indptr, toarray() and tocsr() (tocsr needs SciPy), so that SciPy is not required.",
)
_e("librosa.util.shear", "X | factor axis", "arrays.shear", EQUIVALENT, "SciPy sparse input stays sparse; arrays.SparseRows input gives a SciPy array.")
_e("librosa.util.cyclic_gradient", "data | edge_order axis", "arrays.cyclic_gradient", EQUIVALENT)
_e(
    "librosa.util.fill_off_diagonal",
    "x | radius value",
    "arrays.fill_off_diagonal",
    EQUIVALENT,
    options={"value": Same("None means 0.")},
)
_e("librosa.util.index_to_slice", "idx | idx_min idx_max step pad", "arrays.index_to_slice", EQUIVALENT)
_e("librosa.util.count_unique", "data | axis", "arrays.count_unique", EQUIVALENT)
_e("librosa.util.is_unique", "data | axis", "arrays.is_unique", EQUIVALENT)
_e(
    "librosa.util.interp_broadcast",
    "| x1 x1_pos x2 x2_pos interp_pos op kind fill_value axis",
    "arrays.interp_broadcast",
    WITH_DIFFERENCES,
    "float32 arrays stay in float32 (librosa: float64 for every input); spline orders above 3 are not accepted.",
    options={
        "op": Same("'multiply' is numpy.multiply, which is also accepted."),
        "kind": Same("None means 'linear'."),
    },
)
_e("librosa.util.is_positive_int", "x", "arrays.is_positive_int", EQUIVALENT)
_e("librosa.util.valid_int", "x | cast", "arrays.valid_int", EQUIVALENT)
_e("librosa.util.valid_intervals", "intervals", "arrays.valid_intervals", EQUIVALENT)
_e(
    "librosa.util.example_info",
    "key | html",
    "generation.example_clip_info",
    WITH_DIFFERENCES,
    "Returns one sentence on a synthesised clip rather than printing a recording's licence and details.",
    options={
        "key": Renamed("name", "Clip names differ; see generation.example_clip_names()."),
        "html": Unsupported("the result is plain text."),
    },
)
_e(
    "librosa.util.find_files",
    "directory | ext recurse case_sensitive limit offset",
    "arrays.find_files",
    WITH_DIFFERENCES,
    "case_sensitive=False matches any letter case (librosa: all-lower or all-upper only); directories named like files are left out.",
)
_e(
    "librosa.util.example",
    "key | hq url",
    "generation.example_clip",
    WITH_DIFFERENCES,
    "Returns audio rather than a file path. The clips are synthesised, so they are not librosa's recordings; see generation.example_clip_names().",
    options={
        "key": Renamed("name", "Clip names differ; see generation.example_clip_names()."),
        "hq": Absorbed("choose sample_rate= and dtype= instead."),
        "url": Unsupported("clips are not downloaded."),
    },
)
_e(
    "librosa.util.list_examples",
    "",
    "generation.example_clip_names",
    WITH_DIFFERENCES,
    "Returns a list of names rather than printing a table.",
)
_none(
    "File discovery and citation helpers, and details of librosa's recordings.",
    "librosa.util.cite(version | bib)",
)
_as("librosa.util.ex", "librosa.util.example")
_as("librosa.ex", "librosa.util.example")
_as("librosa.example", "librosa.util.example")
_as("librosa.cite", "librosa.util.cite")
_none(
    "Exception types; errors from this library derive from audio_samples.AudioError and the matching built-in exception.",
    "librosa.LibrosaError()",
    "librosa.ParameterError()",
    "librosa.util.exceptions.LibrosaError()",
    "librosa.util.exceptions.ParameterError()",
)
_none("A caching decorator; this library has no cache.", "librosa.cache(level)")
_none(
    "Version and environment reporting.",
    "librosa.show_versions()",
    "librosa.__version__()",
    "librosa.version.version()",
    "librosa.version.show_versions()",
    "librosa.version.short_version()",
)

# librosa.segment and librosa.sequence --------------------------------------

_e(
    "librosa.segment.recurrence_matrix",
    "data | k width metric sym sparse mode bandwidth self axis full",
    "sequence.recurrence_matrix",
    WITH_DIFFERENCES,
    "sparse=True returns arrays.SparseRows; exact neighbour search. Connectivity results are bool, as librosa's.",
)
_e(
    "librosa.segment.cross_similarity",
    "data data_ref | k metric sparse mode bandwidth full",
    "sequence.cross_similarity",
    WITH_DIFFERENCES,
    "As recurrence_matrix.",
)
_e("librosa.segment.recurrence_to_lag", "rec | pad axis", "sequence.recurrence_to_lag", EQUIVALENT, "Dense input.")
_e("librosa.segment.lag_to_recurrence", "lag | axis", "sequence.lag_to_recurrence", EQUIVALENT, "Dense input.")
_e("librosa.segment.path_enhance", "R n | window max_ratio min_ratio n_filters zero_mean clip **", "segment.path_enhance", EQUIVALENT, "The convolution options are mode, cval and origin.")
_e("librosa.segment.timelag_filter", "function pad index", "segment.timelag_filter", EQUIVALENT, "Returns a callable object; the wrapped function's signature is not copied onto it.")
_e(
    "librosa.segment.agglomerative",
    "data k | clusterer axis",
    "segment.agglomerative",
    EQUIVALENT,
    "No scikit-learn needed; a clusterer object with fit and labels_ is still accepted.",
)
_e("librosa.segment.subsegment", "data frames | n_segments axis", "segment.subsegment", EQUIVALENT)
_e(
    "librosa.sequence.dtw",
    "X Y | C metric step_sizes_sigma weights_add weights_mul subseq backtrack global_constraints band_rad return_steps",
    "sequence.dtw",
    EQUIVALENT,
    "int64 paths.",
)
_e("librosa.sequence.dtw_backtracking", "steps | step_sizes_sigma subseq start", "sequence.dtw_backtracking", EQUIVALENT)
_e(
    "librosa.sequence.rqa",
    "sim | gap_onset gap_extend knight_moves backtrack",
    "sequence.rqa",
    WITH_DIFFERENCES,
    "Same scores; the path through the second column follows the move actually taken (librosa stores a wrong back-pointer there).",
)
_e("librosa.sequence.viterbi", "prob transition | p_init return_logp transition_min_prob", "sequence.viterbi", EQUIVALENT, "int64 states (librosa: uint16).")
_e(
    "librosa.sequence.viterbi_discriminative",
    "prob transition | p_state p_init return_logp transition_min_prob",
    "sequence.viterbi_discriminative",
    EQUIVALENT,
    "As viterbi.",
)
_e(
    "librosa.sequence.viterbi_binary",
    "prob transition | p_state p_init return_logp transition_min_prob",
    "sequence.viterbi_binary",
    WITH_DIFFERENCES,
    "States are bool by default (librosa: uint16); pass dtype=numpy.uint16 for librosa's dtype.",
)
_e("librosa.sequence.transition_uniform", "n_states", "sequence.transition_uniform", EQUIVALENT)
_e("librosa.sequence.transition_loop", "n_states prob", "sequence.transition_loop", EQUIVALENT)
_e("librosa.sequence.transition_cycle", "n_states prob", "sequence.transition_cycle", EQUIVALENT)
_e("librosa.sequence.transition_local", "n_states width | window wrap", "sequence.transition_local", EQUIVALENT)
_e("librosa.sequence.path_to_steps", "path | inverse", "sequence.path_to_steps", EQUIVALENT)

# librosa.display -----------------------------------------------------------

_DISPLAY = "Same capability under this library's names and argument names; returns (fig, ax, artist)."
_e("librosa.display.specshow", "data | **", "mpl.plot_matrix", WITH_DIFFERENCES, _DISPLAY + " vscale is value_scale; cmap_seq and the like are sequential, boolean, diverging, cyclic.")
_e("librosa.display.waveshow", "y | **", "mpl.plot_waveform_adaptive", WITH_DIFFERENCES, _DISPLAY + " Takes AudioSamples or an array with sample_rate.")
_e("librosa.display.AdaptiveWaveplot", "| **", "mpl.AdaptiveWaveform", WITH_DIFFERENCES, "Returned by plot_waveform_adaptive.")
_e("librosa.display.wavebars", "| **", "mpl.plot_waveform_bars", WITH_DIFFERENCES, _DISPLAY)
_e("librosa.display.wavef0", "| **", "mpl.plot_waveform_f0", WITH_DIFFERENCES, _DISPLAY + " method is 'adaptive' or 'bars'.")
_e("librosa.display.multiplot", "| **", "mpl.plot_grid", WITH_DIFFERENCES, "kind is 'waveform', 'bars' or 'matrix'; shared gives one colour range.")
_e("librosa.display.highlight", "| **", "mpl.highlight_effects", EQUIVALENT)
_e("librosa.display.legend_for_axes", "| **", "mpl.legend_for_axes", EQUIVALENT)
_e("librosa.display.colorbar_db", "| **", "mpl.add_db_colorbar", EQUIVALENT)
_e("librosa.display.colorbar_phase", "| **", "mpl.add_phase_colorbar", EQUIVALENT)
_e("librosa.display.infer_cmap", "| **", "mpl.infer_colormap", WITH_DIFFERENCES, "Keywords are sequential, boolean, diverging and threshold.")
_e("librosa.display.TimeFormatter", "| **", "mpl.TimeFormatter", EQUIVALENT)
_e("librosa.display.NoteFormatter", "| **", "mpl.NoteFormatter", EQUIVALENT)
_e("librosa.display.LogHzFormatter", "| **", "mpl.LogHzFormatter", EQUIVALENT)
_e("librosa.display.ChromaFormatter", "| **", "mpl.ChromaFormatter", EQUIVALENT)
_e("librosa.display.ChromaFJSFormatter", "| **", "mpl.ChromaFJSFormatter", EQUIVALENT)
_e("librosa.display.ChromaSvaraFormatter", "| **", "mpl.ChromaSvaraFormatter", EQUIVALENT)
_e("librosa.display.FJSFormatter", "| **", "mpl.FJSFormatter", EQUIVALENT)
_e("librosa.display.TonnetzFormatter", "| **", "mpl.TonnetzFormatter", EQUIVALENT)
_e("librosa.display.Transformf0", "| **", "mpl.F0Transform", WITH_DIFFERENCES, "scale gives the octaves per unit amplitude.")

# ---------------------------------------------------------------------------

# Implementation modules, with the public module their names are re-exported from.
ALIASES: dict[str, str] = {
    "librosa.core": "librosa",
    "librosa.core.audio": "librosa",
    "librosa.core.constantq": "librosa",
    "librosa.core.convert": "librosa",
    "librosa.core.fft": "librosa",
    "librosa.core.harmonic": "librosa",
    "librosa.core.intervals": "librosa",
    "librosa.core.notation": "librosa",
    "librosa.core.pitch": "librosa",
    "librosa.core.spectrum": "librosa",
    "librosa.feature.rhythm": "librosa.feature",
    "librosa.feature.spectral": "librosa.feature",
    "librosa.feature.utils": "librosa.feature",
    "librosa.util.decorators": "librosa.util",
    "librosa.util.deprecation": "librosa.util",
    "librosa.util.files": "librosa.util",
    "librosa.util.matching": "librosa.util",
    "librosa.util.utils": "librosa.util",
}

# Names from older librosa releases. The value is the current name, or a sentence when there
# is none.
REMOVED: dict[str, str] = {
    "librosa.beat.tempo": "librosa.feature.tempo",
    "librosa.feature.rmse": "librosa.feature.rms",
    "librosa.output.write_wav": "write the audio with io.save.",
    "librosa.util.example_audio_file": "librosa.util.example",
    "librosa.display.waveplot": "librosa.display.waveshow",
}
