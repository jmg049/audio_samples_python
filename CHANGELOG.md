# Changelog

All notable changes to the `audio_samples` Python package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package binds the Rust crate
[`audio_samples`](https://github.com/jmg049/audio_samples). Fixes and behaviour
changes in the core are summarised here where Python users will notice them; the
crate's [CHANGELOG](https://github.com/jmg049/audio_samples/blob/main/CHANGELOG.md)
has the full detail.

## [Unreleased]

- **`str(audio)` is one line and `repr(audio)` previews each channel.** `str` gives
  `AudioSamples<f32>: 2 ch × 48000 samples @ 48000 Hz (1.000 s)`; `repr` adds a line per channel
  (up to eight) with its first and last three samples. `str` of mono audio used to include a
  preview line, and the header used `x` for `×`.
- **Onset peak-picking presets are given in seconds and take the sample rate.**
  `PeakPickingConfig.music`, `.speech` and `.drums` were class attributes sized as if their
  lengths counted audio samples, so on an onset strength function they spanned seconds and found
  almost nothing. They are now static methods taking `(sample_rate, hop_size)`, with threshold
  windows of 0.2 s, 0.2 s and 0.1 s and minimum separations of 0.1 s, 80 ms and 30 ms.
  `OnsetDetectionConfig.musical`, `.percussive`, `.speech`, `SpectralFluxConfig.percussive`,
  `.musical` and `ComplexOnsetConfig.percussive`, `.musical`, `.speech` take the sample rate,
  e.g. `OnsetDetectionConfig.percussive(audio.sample_rate)`; the `SpectralFluxConfig` and
  `ComplexOnsetConfig` presets are static methods instead of class attributes.
- **`VadMethod.spectral` works.** It raised an error. A frame is speech when its level reaches
  `energy_threshold_db` and at least `spectral_ratio_threshold` of its energy lies between
  `speech_band_low_hz` and `speech_band_high_hz`; loud noise, hum and hiss are rejected.
- **`LimiterConfig.isp_limiting` detects true (inter-sample) peaks** at 4× oversampling; it had
  no effect. The `LimiterConfig` presets set it, so their output changes for material whose true
  peak exceeds its sample peak.
- **The `SideChainConfig` filters, pre-emphasis and mix are applied** to the detector input when
  the side chain is enabled, in the plain and the side-chain methods; they were ignored.
  `SideChainConfig(enabled=True)` now defaults to `external_mix=1.0` (the side-chain signal
  alone; it was 0.0 but had no effect). `apply_compressor_sidechain` and
  `apply_limiter_sidechain` accept a mono side-chain signal for multi-channel audio, or one with
  the same channel count.

- **`==` and `!=` on AudioSamples compare sample type, sample rate, shape and samples**; `==`
  returned False for any two objects.
- **`forms.resample` and `forms.pitch_shift` with `scale=True` multiply by the reciprocal of the
  square root of the ratio** instead of dividing by it: within one rounding of the quotient, and
  several times cheaper per sample. For float32 and float64 input, `forms.resample` applies it
  while the samples are written (the same values, one pass fewer).
- **Downsampling by ratios such as 22050 to 16000 or 8000 Hz and 48000 to 44100 Hz is faster**
  and its output changes at the stopband level; see the crate's changelog.
- **`features.wavelet_lengths` and `features.wavelet` accept one frequency when `alpha` is
  given.** The bandwidth then does not come from the spacing of the frequencies; without
  `alpha`, two or more are still required.
- **`io.read_and_resample` keeps the file's sample type.** Without `as_type` it returns the
  file's own sample type, as `io.read` and `AudioSamples.resample` do (16-bit files give `int16`);
  it converted to `f32` before. Pass `as_type=SampleType.F32` for the former result.
- **More of librosa's option values work.** `AudioSamples.magphase` takes a fractional or negative
  `power`; `AudioSamples.griffinlim_cqt` takes `init="random"` with `rng`; the `rng` of
  `griffinlim`, `griffinlim_audio`, `griffinlim_cqt` and `inverse` takes a NumPy generator (one seed
  is drawn from it, which advances it); the `features.spectral_*_from_spectrogram` and
  `poly_features_from_spectrogram` take a time-varying 2-D or 3-D `freq`; `onset_strength_multi`
  takes slices as `channels` and `sequence.sync` slices as `idx`; `rhythm.metrogram` takes
  `aggregate=None`; `effects.nn_filter` takes `scipy.sparse` recurrence matrices and any aggregate;
  `trim_db` and `split_db` take any callable as `ref` and `aggregate`; `io.read` and
  `io.StreamedAudioReader` take a binary file object (WAV or FLAC). Callables run with the GIL held.
  `harmonic` / `percussive` still raise for `mask=True`: librosa returns the inverse STFT of the
  mask there, and the masks are `effects.hpss(S, mask=True)`.
- **The `io` stub lists `set_thread_limit` and `thread_limit`** in `__all__`.
- **`io.save` and `io.write_with_options` take `flac_md5`.** `flac_md5=False` writes a FLAC file
  without the MD5 signature of its audio, which encodes faster. `as_type=` now converts block by
  block as the file is written, so saving no longer holds a converted copy of the signal.
- **`io.set_thread_limit` and `io.thread_limit`.** A limit on the threads a file read or decode uses,
  settable at any time, alongside the environment variable `AUDIO_SAMPLES_IO_THREADS`.
- **Streaming reads keep the file's sample type.** `read_frames`, `frames`, `windows` and `samples`
  of `io.StreamedAudioReader` default to the sample type of the file; they converted to `f32`
  before. Pass `dtype=SampleType.F32` for the former result.
- **Import is close to free.** `import audio_samples` no longer loads the
  extension; the first access to one of its names (or `import
  audio_samples.io` and the like) does. Everything else behaves as before.
- **The wheel no longer ships** `logo.png`, benchmark scripts or notebooks. The
  README's logo uses an absolute URL, so it shows on PyPI.
- **Everything else librosa 1.0 computes.** New submodules: `linalg` (`nnls`, `pinv`, `eigh`,
  `svd`); `inverse` (`mel_to_stft`, `mel_to_audio`, `mfcc_to_mel`, `mfcc_to_audio`);
  `factorization` (`Nmf`, `decompose`); `pitch_filterbank` (`mr_frequencies`,
  `semitone_filterbank`, `iirt`), with `AudioSamples.iirt`; `segment` (`diagonal_filter`,
  `path_enhance`, `timelag_filter`, `agglomerative`, `subsegment`); `notation` (tuning systems,
  Functional Just System, svara, mela and thaat names, `midi_to_note`, `hz_to_note`, `note_to_midi`,
  `note_to_hz`); `arrays` (local extrema, event and interval matching, one-dimensional
  interpolation, padding, reordering, compressed sparse rows, argument checks, `zero_crossings`,
  `tone`, `chirp`). Also `spectral.mellin_transform`, `sequence.path_to_steps`,
  `utils.blocks_to_{frames,samples,time}` and `generation.example_clip_info`.
- **Array forms and options.** `audio_samples.forms` holds array forms of methods, for options an
  `AudioSamples` makes unnecessary or that need another input: `to_mono`, `to_stereo`, `to_multi`,
  `get_duration`, `read_segment`, `stream_blocks`, `resample`, `pitch_shift`, `frame`, `onset_detect`,
  `beat_track`, `stft`, `istft`, `reassigned_spectrogram`, `chroma_cqt`, `chroma_cens`, `chroma_vqt`,
  `tempogram_ratio`, `onset_strength` and `salience`. `AudioSamples.vqt` takes `intervals`;
  `interp_harmonics` and `f0_harmonics` take `kind` and `axis`; `recurrence_matrix` and
  `cross_similarity` take `sparse=True` and `recurrence_matrix` takes `axis`; `cq_to_chroma` takes
  its smoothing `window`; `wavelet` and `wavelet_lengths` take `alpha`.
- **Display.** `mpl.plot_matrix` with typed axes (time, lag, frequency, mel, constant-Q, note, svara,
  FJS, chroma, tempo, tonnetz) and dB and phase value scales; `mpl.plot_waveform_adaptive`,
  `mpl.plot_waveform_bars`, `mpl.plot_waveform_f0`, `mpl.plot_grid`, `mpl.infer_colormap`, dB and phase
  colorbars, a shared legend, highlights and the tick formatters.
- **Unary operators.** `-audio`, `+audio` and `abs(audio)` return audio of the same sample type. Integer results saturate,
  and `uint8` works relative to silence.
- **Arithmetic keeps the sample type.** `1.0 + audio` returned float64; `audio / other` and `audio / array` returned
  float64 for integer audio, `/=` changed the sample type of the audio it was applied to, and an array of another dtype did the
  same for `+`, `-` and `*`. All of them now keep the sample type of the audio on the left, and audio of another sample
  type is converted to it (this panicked before). Integer results saturate instead of wrapping, `u8` audio respects
  its offset, and an integer divided by zero gives the limit with its sign. On integer audio a float scalar is in the
  floating-point range, so `audio + 1.0` adds full scale, and it must be within -1.0 to 1.0 or `ValueError` is raised. An int
  or NumPy integer scalar is a number of sample steps relative to silence, so `audio + 1` adds one step on every integer
  type, on either side of the operator, and `OverflowError` is raised if it does not fit the type (`audio + 40000` on int16,
  `audio + 200` on uint8, whose range is -128 to 127). A NumPy `uint8` scalar on `uint8` audio is a sample, so
  `audio + np.uint8(128)` changes nothing. `*` and `/` are gains, so `audio * 2` equals `audio * 2.0`, and only the result
  saturates. Non-finite scalars raise `ValueError` on integer audio, a non-finite array too, and dividing integer audio
  by a zero scalar raises `ZeroDivisionError`. Dividing by an array or other audio works sample by sample: a divisor sample
  of silence gives the limit with the sign of the dividend, and silence over silence gives silence (128 for `uint8`). Arithmetic with NumPy on the left
  (`np.int16(1) + audio`, `array * audio`) now follows the same rules. On float audio an int is the same number as the float. A bool scalar raises `TypeError`.
- **Type stubs are checked and precise.** `mypy.stubtest`, mypy and pyright run in pre-commit and in CI. `AudioSamples`
  is generic in its sample type (`AudioSamples[numpy.float32]` also works at run time), and overloads give the dtype of
  analysis results from the sample type or the input array, the bool results of `pyin`, `recurrence_matrix` and
  `viterbi_binary`, and the results selected by `units=`, `mode=`, `return_zf=`, `backtrack=` and similar arguments. An
  input of unknown type gives a result of unknown type. Corrected signatures include `apply_compressor`, `apply_limiter`
  and `normalize`; `scale`, `clip`, `normalize` and `remove_dc_offset` return `None`. The stubs for `utils` move to
  `audio_samples/utils/`; `audio_samples.audio_math` and `audio_samples.detection` never existed at run time and are gone
  from the stubs, with their contents under `audio_samples.utils`.
- **Migration checker.** `python -m audio_samples.migrate PATH [PATH ...]` lists the librosa calls in Python
  files and notebooks with the equivalent, how closely it matches and what to watch for in that call. It has
  JSON output, a summary and an exit status for CI, and it never changes the files.
- **Example clips.** `generation.example_clip` renders named clips of music and speech with known
  contents at any sample rate, duration and sample type. Nothing is stored or downloaded, and the same
  arguments give the same bytes on any platform with IEEE 754 arithmetic.
- **Free-threaded Python.** The module states `gil_used = false`, so on `python3.14t` the
  GIL stays off. A read that meets a concurrent
  `SosFilter.process_samples_in_place` on the same array raises `TypeError` instead of
  a `PanicException`. See the "Threads and free-threaded Python" page.
- **Threads.** Streaming readers and writers work from any thread, one call at a time; they
  no longer raise a `PanicException` on another thread. On a free-threaded build, an in-place
  method on audio built on a NumPy array and reads of that array raise `TypeError` when they meet.
- **Faster.** `hpss` on a spectrogram (median filter), `melspectrogram`, `mfcc_frames` and `to_mono`
  of decoded stereo. Results are unchanged.
- **Example gallery.** The docs have a gallery by task (`docs/gallery`, built with
  sphinx-gallery): each entry runs during the docs build, and a failure stops it.
- `AudioSamples.apply_compressor` and `apply_limiter` stubs match the methods, which take a
  config object.
- **librosa parity.** A librosa 1.0-compatible API alongside the existing one:
  librosa-style methods on `AudioSamples` and the new submodules
  `audio_samples.spectral`, `features`, `sequence`, `rhythm` and `effects`.
  They take librosa's keyword names, return `(channels, F, T)` arrays, and
  are tested against librosa 1.0 golden fixtures. The
  docs gain a librosa migration guide that maps every librosa function to its
  equivalent and lists every difference.
- **Output precision follows the input.** The librosa-compatible
  `AudioSamples` methods return float32 / complex64 arrays for uint8, int16,
  int24 and float32 audio, and float64 / complex128 arrays for int32 and
  float64 audio (the float type that holds every sample value exactly).
  The free functions in `spectral`, `features`, `rhythm`, `effects` and
  `sequence` compute in float32 for float32 / complex64 arrays and in float64
  for any other input. Filterbanks, windows, weightings, tempo and tuning stay
  float64. Audio-returning methods (`time_stretch`, `pitch_shift`,
  `preemphasis`, `trim_db`, `harmonic`, ...) return the receiver's sample
  type. `istft_matrix`, `icqt`, `griffinlim_cqt` and `griffinlim_audio` take a
  `dtype=` sample type; without it they return float32 audio for a
  single-precision input matrix and float64 audio otherwise.
- **Defaults that differ from librosa.** Where librosa's default changes the
  data without being asked, the bindings use a different default; pass
  librosa's value to reproduce a librosa call: `top_db=None` in
  `power_to_db`, `amplitude_to_db`, `mfcc_frames` and `perceptual_weighting`
  (librosa: 80); `min_db=None` in `frequency_weighting` and
  `multi_frequency_weighting` (librosa: -80); `tuning=0.0` in `chroma_stft`,
  `chroma_cqt`, `chroma_cens`, `tonnetz` and `chroma_stft_from_spectrogram`
  (librosa: None, which estimates it); `trim=False` in `beat_track` and
  `beat_track_with_bpm` (librosa: True); `rng=0` in `griffinlim` and
  `griffinlim_audio` (librosa: unseeded); `align_zeros=False` in `remix`
  (librosa: True).
- **Required arguments.** `sr` is required in every free function where
  librosa defaults it to 22050, and the `utils` frame and time conversions
  require `sample_rate` or `sr` (they used to fall back to 22050).
  `hop_length` is required in `pcen`, `samples_to_frames`,
  `frames_to_samples`, `time_to_frames_floor`, `times_like`, `samples_like`,
  and in `clicks` when `frames=` is given (librosa: 512). `top_db` is required
  in `trim_db`, `trim_db_in_place` and `split_db` (librosa: 60).
- **Teaching material.** An explain API (`audio_samples.explain`), 14 tutorial
  notebooks, a page for instructors and a set of usability fixes for students.

### Added

- Explain API: `audio_samples.explain.explain(obj, *, sample_rate, n_samples,
  n_channels, f0_range, n_fft, n_mels)` explains `StftParams`,
  `SpectrogramParams`, `MelParams`, `MfccParams`, `WindowType`,
  `IirFilterDesign`, `EqBand`, `CompressorConfig`, `LimiterConfig`, `VadConfig`
  and `OnsetDetectionConfig` from your settings, before anything runs. Topic
  functions cover sampling, aliasing, quantisation, the DFT, windows, YIN,
  autocorrelation pitch, LPC, the cepstrum, resampling and FIR filters.
  `AudioSamples.explain(obj, ...)` uses the signal's rate, length and channels,
  and the config classes gain `.explain(...)`.
- `ExplainDoc`: summary, formulas, symbols, derived quantities, tables,
  trade-offs and warnings as plain Python data, with `text()`, `markdown()`,
  `html()`, `quantity()` and `has_warning()`. It renders in Jupyter with maths.
- Tutorials: 14 notebooks (`docs/source/tutorials/`), from sampling to codecs,
  checked against NumPy, SciPy and librosa. Solution cells are tagged so that
  student versions can be generated. There is also a "For instructors" page.
- `AudioSamples` librosa-style spectrum methods: `stft_matrix`, `istft_matrix`
  (static), `spectrogram_matrix`, `melspectrogram` and `mfcc_frames`.
- `AudioSamples` librosa-style features: `spectral_{centroid,bandwidth,rolloff,flatness,contrast}_frames`,
  `poly_features`, `rms_frames`, `zero_crossing_rate_frames` and
  `reassigned_spectrogram`.
- `AudioSamples` tonal, constant-Q and pitch methods: `chroma_stft`,
  `piptrack`, `estimate_tuning`, `cqt`, `vqt`, `pseudo_cqt`, `hybrid_cqt`,
  static `icqt` and `griffinlim_cqt`, `chroma_cqt`, `chroma_cens`, `chroma_vqt`,
  `tonnetz`, `yin` and `pyin`.
- `AudioSamples` rhythm methods: `onset_strength`, `onset_strength_multi`,
  `onset_detect`, `tempogram`, `fourier_tempogram`, `tempo`, `beat_track`
  (returns `(tempo, beats)`), `plp`, `tempogram_ratio` and `hybrid_tempogram`.
- `AudioSamples` effects, each audio-to-audio one also available as an
  `_in_place` variant: `time_stretch`, `pitch_shift`, `preemphasis`,
  `deemphasis`, `trim_db`, `split_db`, `remix`, `harmonic`, `percussive`,
  `harmonic_percussive`, `lpc` and the static `griffinlim_audio`.
- `audio_samples.spectral`: `stft`, `istft`, dB conversions, `mel_filterbank`,
  `get_window`, `window_sumsquare`, `window_bandwidth`, `delta`,
  `stack_memory`, `frame`, `melspectrogram_from_spectrogram` and
  `mfcc_from_mel_db`.
- `audio_samples.features`: the `*_from_spectrogram` descriptors, `pcen`,
  frequency weightings (A to D, Z, multi, perceptual), `chroma_filterbank`,
  `normalize`, tuning estimation, `cq_to_chroma`, `wavelet`,
  `wavelet_lengths`, CENS and tonnetz post-processing, `yin`, `pyin`,
  `salience`, `interp_harmonics` and `f0_harmonics`.
- `audio_samples.sequence`: `dtw`, the Viterbi decoders, transition matrices,
  `rqa`, `recurrence_matrix`, `cross_similarity`, the recurrence/lag
  conversions and `sync`.
- `audio_samples.rhythm`: array-input onset, tempo and beat functions,
  `onset_backtrack`, `peak_pick`, tempo frequencies and `metrogram`.
  `aggregate` accepts NumPy reductions, names or any callable, and `prior`
  accepts a `scipy.stats` distribution or a callable.
- `audio_samples.effects`: `phase_vocoder`, `preemphasis`/`deemphasis` with
  filter state, `softmask`, `hpss` (librosa `decompose.hpss`), `nn_filter`,
  `griffinlim`, `lpc` and `clicks`.
- `audio_samples.utils`: librosa unit conversions (`time_to_samples`,
  `samples_to_frames`, `frames_to_samples`, `time_to_frames_floor`,
  `times_like`, `samples_like`, `hz_to_octs`, `octs_to_hz`, `a4_to_tuning`,
  `tuning_to_a4`, `cqt_frequencies`, `key_to_notes`, `key_to_degrees`),
  `mu_law_encode` and `mu_law_decode` (also at top level). There is also a
  `htk=` keyword on the mel helpers and `n_fft=` on `frames_to_time`.
- `IirFilterDesign`: `passband_ripple`, `stopband_attenuation` and
  `bessel_norm` constructor arguments; Chebyshev I/II, elliptic, Bessel and
  Butterworth band-stop classmethods; `sos(sample_rate)` returns SciPy's
  `(n, 6)` SOS array; `frequency_response(freqs, sample_rate)`.
- `SosFilter.from_sos(sos, fs)`, `SosFilter.sos()` and `SosFilter.filtfilt(x)`.
- `AudioSamples.apply_filter` / `apply_filter_in_place` (FIR "valid"
  convolution, like `np.convolve(..., "valid")`).
- `audio_samples.exceptions`: `AudioError` and subclasses that carry the Rust
  diagnostic's `.code` and `.help`. Each subclass also inherits the built-in
  exception raised before (`ValueError`, `TypeError`, ...), so existing
  `except` clauses keep working.
- The spectrogram parameter classes (`StftParams`, `SpectrogramParams`,
  `MelParams`, `MfccParams`, `LogParams`, ...) are re-exported at top level,
  and `audio_samples.spectrograms` is importable. `ChromaNorm` is exported.
- `AudioSamples._repr_html_` shows a summary table and an audio player in
  Jupyter. Config classes have field-listing reprs.
- `EnvelopeFollower.process`, `process_array` and `reset`; read-only getters
  on `CompressorConfig`, `VadConfig` and
  `PeakPickingConfig.min_peak_separation`.
- NumPy interop: ufuncs and array functions (`np.sin`, `np.mean`, `np.max`,
  `np.fft.rfft`, ...) accept `AudioSamples`, and `__array__` accepts `copy=`.
- Docs: a librosa migration guide, reference pages for the new modules, and a
  "Coming from librosa" section in the README.

### Changed

- `sequence.viterbi_binary` returns its states as `bool` by default (one byte
  per entry; librosa returns `uint16`). `dtype=numpy.uint16` or
  `dtype=numpy.int64` gives an integer array.
- `AudioSamples.apply_window` multiplies by float64 coefficients and rounds
  integer results to nearest, saturating at the type's range. Integer audio
  used to quantise the window to the sample type, so fractional windows
  zeroed the signal (or overflowed). Float audio is unchanged.
  `apply_filter` / `apply_filter_in_place` likewise filter integer audio with
  float64 taps on its raw scale, so taps outside `[-1, 1]` work.
- Behaviour changes from the core crate (see its changelog):
  - Integer audio: `normalize` with the mean or median method,
    `remove_dc_offset`, `to_mono` (average), `to_stereo` (pan), `pan` and
    `balance` round to nearest and saturate (24-bit values no longer wrap;
    uint8 pans around 128). `to_mono` with the weighted and centre methods no
    longer returns silence for integer audio. Z-score normalisation of
    integer audio raises instead of returning near-silence.
  - `zero_crossing_rate` on multichannel audio is the mean of the per-channel
    rates. It used to sum the channels' crossings, so two identical channels
    gave twice the mono rate.
  - Resampling (`resample`, `resample_by_ratio`, `io.read_and_resample`) uses
    a delay-free windowed-sinc resampler. The output is aligned with the input
    and is exactly `ceil(n * out / in)` samples long. `ResamplingQuality.fast`
    downsampling is now anti-aliased, so it is no longer identical to
    `x[::2]`.
  - `mu_compress` follows the standard curve `ln(1 + mu|x|) / ln(1 + mu)`.
  - `PeakPickingConfig.default()` (used by the default onset detection
    config) uses a 9-frame delta threshold (0.07) with an absolute floor of
    0.1, a 3-frame minimum separation and no median filter, so it finds the
    onsets of a clean track. The `music`/`speech`/`drums` presets are
    unchanged.
  - Filter designs match `scipy.signal`. High-pass Butterworth/Chebyshev
    filters of order other than 2 were frequency-mirrored, odd orders had a
    doubled pole, and the simple filters were unstable or silent on int16
    audio.
- `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `frames_to_time` and
  `time_to_frames` accept arrays. Scalar input still returns a scalar.
- `aus.WindowType` is now the class that `StftParams` accepts. The old enum
  is still available as `aus.types.WindowType`.
- Arithmetic, `convolve`, `deconvolve`, `cross_correlation`, `mix`,
  `concatenate`, `stack` and spectrogram parameters check that sample rates
  match, and the error message names both rates.
- Empty buffers are rejected when the `AudioSamples` is constructed, instead
  of causing a `PanicException` later.
- `AudioSamples.copy()` makes a deep copy of NumPy-backed data. The copy used
  to share the buffer, so filtering the copy also filtered the original.
- In-place operations on read-only NumPy buffers copy the data first instead
  of writing through it.
- `new_multi` / `from_array` warn when an array looks like
  `(samples, channels)`. Under `-W error` the warning raises.
- Filter responses accept the strings `'lowpass'`, `'highpass'`, `'bandpass'`
  and `'bandstop'`, and filter errors raise `ValueError` (some raised
  `TypeError`).
- `hpss` accepts an `HpssConfig`. Generators accept any `dtype` that
  `numpy.dtype()` accepts; the default is still float32.
- `resample` and `resample_by_ratio` accept the quality names `"fast"`,
  `"medium"` and `"high"`.
- `inspect.signature` and `help()` work for every public callable.
- Missing files raise `FileNotFoundError`.
- `from audio_samples.spectral|sequence|utils|types|io import ...` works.
- `AudioSamples` is no longer marked unsendable. The package loads the
  extension that was built for the running interpreter when several are
  present.
- The heavy pure-Rust computations release the GIL (for example `pyin`,
  `stft`, `cqt`, `time_stretch` and `hpss`), so they run in parallel across
  threads.
- The `spectrograms` pip dependency was dropped. Its objects were rejected at
  runtime, and the classes are compiled into the extension.
- The extension is built with the core's `educational` feature, which the
  explain API needs.

### Deprecated

- `AudioSamples.frequency_response` emits a `DeprecationWarning`. It always
  returned a flat response.
- `hpss(stft_params=...)` is accepted as a deprecated alias of `config=`.
- `effects.phase_vocoder(hop_length=, n_fft=)` is accepted and ignored with a
  `FutureWarning`, as in librosa.

### Fixed

- `AdaptiveThresholdConfig.min_threshold` / `max_threshold` are readable
  properties (they were write-only, and the stub declared nonexistent
  `set_min_threshold()` / `set_max_threshold()` methods).
- The docs site reports the package version from `Cargo.toml` instead of a
  stale `0.1.1`.
- `np.asarray` on 24-bit audio read freed memory, which corrupted the data in
  about half of the calls.
- Integer arguments that are out of range raise `ValueError` instead of
  `OverflowError` or aborting the interpreter on an impossible allocation (for
  example `midi_to_note(-1)`, `resample(-n)` and `stft(n_fft=2**40)`).
- `effects.lpc` checks the order against the signal length before allocating.
- Nested Python callbacks (`aggregate`/`prior`) no longer clobber the outer
  call's callback and return NaN silently.
- `str()` of multichannel audio no longer panics, and `time_axis()` works on
  multichannel audio.
- Docstrings and stubs were corrected to match the runtime (tested in both
  directions). `onset_detection_function` and `spectral_flux` return
  `(times, values)`. `zero_crossing_rate` is in crossings per second and
  averaged over channels. `detect_beats` uses a fixed-tempo grid.
  `AudioSamples.dtype` is a string. The generator default dtype is float32.
  The resampler docs describe the actual filters and output length. The YIN
  threshold docstring had its direction backwards. README examples are now
  executed by the tests.

## [0.2.4] - 2026-07-08

- Version bump only; no code changes from 0.2.3.

## [0.2.3] - 2026-07-08

### Changed

- Migrated to `audio_samples` 2.0 and `audio_samples_io` 0.3, with all of the
  2.0 breaking changes. These include the `*_in_place` renames, structured
  return types, config structs and `ChannelReduction`.

### Added

- Bindings for the 2.0 functionality. This covers spectral descriptors, PSD,
  `convolve`/`deconvolve`, the constant-Q transform, Chebyshev II, elliptic
  and Bessel IIR designs, `filtfilt`, a streaming `SosFilter`, a parametric
  EQ, gate/expander and sidechain dynamics, pitch contours, key and tempo
  estimation, iterators, new generators, comparison metrics, and WAV/FLAC
  streaming IO with metadata.
- Citation information in the README.

## [0.2.2] - 2026-04-29

### Added

- FLAC support and faster IO, through `audio_samples_io` 0.2 (with
  `audio_samples` 1.0.6).

## [0.2.1] - 2026-03-31

### Changed

- Updated `audio_samples_io` to 0.1.9. Added IO benchmarks.

## [0.2.0] - 2026-03-17

### Changed

- Updated to `audio_samples` 1.0 and pyo3 0.28. Python 3.10 or newer is
  required.
- Plotting uses a matplotlib-based API (the `mpl` extra) by default. The
  Rust plotly backend is opt-in at build time (the `plotting` extra with the
  `audio-samples-plotting` feature).
- Stubs and bindings are split into modules that mirror the core crate.

### Added

- Benchmarks and tested README snippets.

## [0.1.1] - 2026-01-04

### Added

- Documentation site, contributing guide, README and license.

## [0.1.0] - 2026-01-04

### Added

- Initial Python bindings for `audio_samples` and `audio_samples_io`. These
  cover generation, IO, processing, spectral analysis and statistics, with a
  DTMF example.

[Unreleased]: https://github.com/jmg049/audio_samples_python/compare/v0.2.3...HEAD
[0.2.4]: https://pypi.org/project/audio_samples/0.2.4/
[0.2.3]: https://github.com/jmg049/audio_samples_python/releases/tag/v0.2.3
[0.2.2]: https://github.com/jmg049/audio_samples_python/releases/tag/v0.2.2
[0.2.1]: https://pypi.org/project/audio_samples/0.2.1/
[0.2.0]: https://pypi.org/project/audio_samples/0.2.0/
[0.1.1]: https://pypi.org/project/audio_samples/0.1.1/
[0.1.0]: https://pypi.org/project/audio_samples/0.1.0/
