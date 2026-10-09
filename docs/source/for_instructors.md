# For instructors

This page is for lecturers and teaching assistants considering `audio_samples`
for a speech and audio processing course. It covers what the library offers a
course, how the tutorials are organised and checked, and how to verify results
against SciPy and librosa.

## What the library gives a course

- **One object that carries its own metadata.** An `AudioSamples` holds the
  samples, the sample rate and the channel layout together. Arithmetic between
  signals with different sample rates raises an error instead of silently
  producing nonsense.
- **The same core as the Rust crate.** The Python package is a thin binding
  over the Rust `audio_samples` crate, so a notebook and a Rust lab compute the
  same numbers with the same code.
- **Errors that teach.** Invalid parameters raise `audio_samples.exceptions`
  classes (for example `AudioParameterError`, which is also a `ValueError`)
  that carry `.code` and `.help`; the help text appears in the traceback.
- **A library that explains itself.** A configuration can describe what it will
  do before it runs: the formula the code evaluates, every symbol with its unit,
  derived resolution and latency numbers, and warnings about suspicious
  settings. In Jupyter the explanation renders as Markdown with maths.

  ```python
  import audio_samples as aus

  audio = aus.generation.sine_wave(120.0, 0.5, 16000)
  params = aus.StftParams(n_fft=256, hop_size=128, window=aus.WindowType.hanning, centre=True)
  audio.explain(params, f0_range=(80.0, 300.0))   # displays the explanation
  ```

## How the tutorials are organised

The {doc}`tutorials <tutorials/index>` are fourteen notebooks, one per topic of a
typical module, from sampling and quantisation to codecs. Each follows the same
pattern:

1. **Concept**: the theory, with every symbol and unit defined.
2. **With the library**: the API that does it.
3. **Under the hood**: the same result re-derived with NumPy (and SciPy or
   librosa where the conventions match), checked with `assert`.
4. **Ask the library to explain**.
5. **Exercises**, each followed by a solution cell that computes and asserts
   the answer. Remove the cells tagged `solution` to make a student version.

The same course exists in Rust (the crate's `guide` module and its
`examples/tutorial_NN_*.rs` labs), with the same numbers.

### Everything is checked

Every number stated in a notebook is computed and asserted in a code cell, and
the notebooks are executed headlessly in the test suite:

```bash
pytest --nbmake docs/source/tutorials
```

If you adapt a notebook, keep this property: change the number in the code and
the test tells you whether the text is now wrong.

## Suggested labs

| Lab | Topic | Students should leave able to |
|---|---|---|
| 01 | Sampling and aliasing | predict where a tone above Nyquist lands; explain why decimation needs a low-pass filter |
| 02 | Quantisation and dither | derive and measure SQNR ≈ 6.02 b + 1.76 dB; show what TPDF dither does to quiet signals |
| 03 | DFT, windows and leakage | map bins to hertz; choose a window from its main-lobe width and side-lobe level |
| 04 | STFT and spectrograms | trade time resolution against frequency resolution; compute frame counts; read a mel spectrogram |
| 05 | Convolution and FIR filters | relate full and valid convolution lengths; design a windowed-sinc low-pass; measure group delay |
| 06 | IIR filters | design a Butterworth filter, find its poles and zeros, judge stability, measure −3 dB at the cutoff |
| 07 | Resampling | resample without aliasing; measure the alias rejection of each quality tier |
| 08 | LPC and formants | fit an all-pole model with Levinson–Durbin and read formants from its roots |
| 09 | Pitch | estimate f0 with autocorrelation and YIN; explain octave errors |
| 10 | Cepstrum, mel and MFCC | compute a cepstrum and an MFCC pipeline step by step |
| 11 | Energy and VAD | build an energy and zero-crossing voice activity detector with hangover |
| 12 | Dynamics | draw a compressor's static curve and measure attack and release time constants |
| 13 | Onsets and tempo | compute spectral flux, pick peaks and estimate tempo |
| 14 | Psychoacoustics and codecs | convert between perceptual scales; implement μ-law companding |

## Verifying results against SciPy and librosa

The IIR filters are checked against SciPy golden fixtures in the Rust test
suite. For other topics, compare with these conventions in mind:

| Topic | `audio_samples` | Reference default | To compare like with like |
|---|---|---|---|
| STFT window | symmetric Hann (`WindowType.hanning`) | librosa and SciPy: periodic Hann | pass `scipy.signal.windows.hann(N, sym=True)` as librosa's `window=` |
| STFT padding | `centre=True` pads ⌊N/2⌋ zeros at each end | librosa ≥ 0.10: `pad_mode="constant"` | same |
| `rfft` | magnitudes \|X\[k\]\| | `numpy.fft.rfft`: complex spectrum | `np.abs(np.fft.rfft(x))` |
| Mel scale | mel spectrogram: Slaney, no filter normalisation; `utils.hz_to_mel`: HTK | librosa default: Slaney with `norm="slaney"` | `librosa.filters.mel(..., norm=None)`; `librosa.hz_to_mel(f, htk=True)` |
| dB | absolute floor re 1 | librosa: `top_db=80` relative to the maximum | `top_db=None`, same `amin` |
| MFCC | unnormalised DCT-II, lifter 22 | librosa: DCT-II `norm="ortho"`, lifter 0 | rescale by the DCT normalisation; set the lifter |
| Zero-crossing rate | crossings per second over the whole signal | librosa: fraction of samples per frame | divide by the sample rate; frame the signal |
| Integer to float | asymmetric: −1.0 ↔ −32768, +1.0 ↔ 32767 | soundfile: divide by 32768 | differs by at most one least significant bit |
| YIN | de Cheveigné and Kawahara with parabolic interpolation; `detect_pitch_yin` is clip-wide | `librosa.yin`: framewise | compare with `track_pitch` |

Known limitations worth telling students about: in this release `mu_compress`
deviates from the standard μ-law curve, and the gammatone filterbank's default
centre spacing is linear in Hz rather than on the ERB-number scale.

## Setting up a course environment

```bash
pip install audio_samples matplotlib scipy librosa jupyter
```
