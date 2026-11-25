# For instructors

This page is for lecturers and teaching assistants considering `audio_samples`
for a speech and audio processing course. It covers what the library offers a
course, how the tutorials are organised and checked, and how to verify results
against SciPy and librosa.

## What the library gives a course

- **One object that carries its own metadata.** An `AudioSamples` holds the
  samples, the sample rate and the channel layout together. Arithmetic between
  signals with different sample rates raises an error instead of silently
  producing incorrect results.
- **The same core as the Rust crate.** The Python package is a thin binding
  over the Rust `audio_samples` crate, so a notebook and a Rust lab compute the
  same numbers with the same code.
- **Errors with codes and help text.** Most invalid parameters raise `audio_samples.exceptions`
  classes (for example `AudioParameterError`, which is also a `ValueError`)
  that carry `.code` and `.help`; the help text appears in the traceback. A few
  constructors (for example `StftParams(n_fft=0)`) raise a plain `ValueError`
  without them.
- **Configurations that explain themselves.** A configuration can describe what
  it will do before it runs: the formula the code evaluates, every symbol with its unit,
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
3. **How it works**: the same result re-derived with NumPy (and SciPy or
   librosa where the conventions match), checked with `assert`.
4. **Ask the library to explain**.
5. **Exercises**, each followed by a solution cell that computes and asserts
   the answer.
6. **Common pitfalls**, **See also** and **References**.

The same course exists in Rust (the crate's `guide` module, its
`examples/tutorial_NN_*.rs` labs and an illustrated
[book](https://jmg049.github.io/audio_samples/) with figures
computed by the library), with the same numbers wherever the two APIs
cover the same function. Where the Python bindings lack a function (Bark, ERB,
threshold of hearing, codecs, cepstrum, FIR design, autocorrelation-method LPC),
the notebook computes it with NumPy from the same published formula instead of
calling the library.

### Making a student version

Every solution cell is tagged `solution`. Strip them with nbconvert to make
student copies:

```bash
jupyter nbconvert --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags solution \
    --to notebook --output-dir student/ docs/source/tutorials/*.ipynb
```

The student copies still run from top to bottom, but nothing checks the
students' `answer = None  # TODO` cells; keep the full notebooks (or their
solution cells) as the marking key.

### Everything is checked

Every number stated in a notebook's text is meant to be computed and asserted
in a code cell, and the notebooks are executed headlessly in the test suite:

```bash
pytest --nbmake docs/source/tutorials
```

If you adapt a notebook, keep this property: when you change a number in the
code, the test reports whether the text is still correct.

## Suggested labs

Each notebook is sized for a two-hour lab session, like the Rust labs. Executed
headlessly, each runs in well under a minute, so students spend the time on the
exercises rather than waiting for code. The notebooks build on each other in
order; 01–04 form a self-contained block on sampling and spectral analysis.

**Prerequisites.** Python and NumPy (array slicing, broadcasting); complex
numbers and complex exponentials; logarithms and decibels. The DFT is defined
in notebook 03, but students who have met it before move faster. Notebooks 05,
06 and 08 use the z-transform (poles, zeros and transfer functions); notebook 14
assumes notebook 02. No Rust is needed.

| Lab | Topic | Students should leave able to |
|---|---|---|
| 01 | Sampling and aliasing | predict the alias frequency of a tone above Nyquist; explain why decimation needs a low-pass filter |
| 02 | Quantisation and dither | derive and measure SQNR ≈ 6.02 b + 1.76 dB; show what TPDF dither does to quiet signals |
| 03 | DFT, windows and leakage | map bins to hertz; choose a window from its main-lobe width and side-lobe level |
| 04 | STFT and spectrograms | trade time resolution against frequency resolution; compute frame counts; invert an STFT |
| 05 | Convolution and FIR filters | relate full and valid convolution lengths; design a windowed-sinc low-pass; measure group delay |
| 06 | IIR filters | design a Butterworth filter, find its poles and zeros, judge stability, measure −3 dB at the cutoff; design a hum notch |
| 07 | Resampling | resample without aliasing; measure the alias rejection of each quality tier |
| 08 | LPC and formants | fit an all-pole model with Levinson–Durbin and read formants from its roots |
| 09 | Pitch | estimate f0 with autocorrelation and YIN; explain octave errors |
| 10 | Cepstrum, mel and MFCC | compute a cepstrum, read a mel spectrogram and build an MFCC pipeline step by step |
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
| YIN | de Cheveigné and Kawahara's difference function (summed over all N − τ overlapping pairs) and CMND, with parabolic interpolation of the CMND (the paper takes the period from the raw difference function and adds a "best local estimate" step); no voiced estimate when no lag is below the threshold; `detect_pitch_yin` is clip-wide | `librosa.yin`: framewise, sums the difference function over a fixed window of half the frame, falls back to the global minimum | compare with `track_pitch` |

## Known limitations

Students should know about these (the notebooks point them out where they are relevant):

- `mel_spectrogram` without `LogParams` returns the square root of the mel
  power, not the power (notebook 10).
- The default `OnsetDetectionConfig` finds every click of a clean click
  track, but its threshold floor (0.1 of the strongest onset) drops much
  weaker onsets: on notebook 13's eight-click track it finds 5. Notebook 13
  shows this and checks the configuration with `.explain()`.
- `estimate_tempo` makes octave errors (and third-of-tempo errors) on simple
  click tracks; `BeatTrackingConfig.tempo_bpm` is not used by it.
- The Python bindings have no Bark, ERB, threshold-of-hearing, codec,
  cepstrum or FIR-design functions (the Rust crate has most of them), so
  notebooks 05, 10 and 14 compute these with NumPy. Their LPC,
  `AudioSamples.lpc`, uses Burg's method; notebook 08 writes the
  autocorrelation method (Levinson-Durbin) in NumPy and compares the two.
- All signals in the notebooks are synthetic (tones, chirps, noise, click
  tracks and synthetic vowels) and nothing is played back; add a recording of
  your own (and `IPython.display.Audio`) if you want real speech.

## Setting up a course environment

The course needs a build of `audio_samples` that includes the explain API
(`audio_samples.explain`), which is in 0.3.0 and later. Until 0.3.0 is on
PyPI, build the package from source: clone the repository,
create a virtual environment, `pip install maturin`, and run
`maturin develop --release` in the clone. Then install the notebook
dependencies (Python ≥ 3.10; the notebooks need librosa ≥ 0.10):

```bash
pip install "audio_samples" matplotlib scipy librosa jupyter pytest nbmake
```

(Once the release is out, this one command installs everything.) Check the
installation by running every notebook headlessly:

```bash
pytest --nbmake docs/source/tutorials
```
