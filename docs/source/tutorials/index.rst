Tutorials: a short course in speech and audio processing
=========================================================

Fourteen self-checking notebooks, one per topic of a typical speech and audio
processing module. Each starts from a concept, shows it with ``audio_samples``,
re-derives the result from first principles with NumPy ("under the hood"), asks
the library to explain its own settings, and ends with exercises whose answers
are computed and asserted, so every number in the text is checked whenever the
notebook runs.

The notebooks need a build of ``audio_samples`` that includes the explain API
(``audio_samples.explain``): the release after 0.2.4. Until that is on PyPI, build
the package from source with ``maturin develop --release``. Then install the
notebook dependencies::

    pip install matplotlib scipy "librosa>=0.10" jupyter pytest nbmake

Run them locally with Jupyter, or check them all headlessly with::

    pytest --nbmake docs/source/tutorials

Each exercise is followed by a solution cell tagged ``solution``. To make a
student version without the solutions, strip those cells with nbconvert::

    jupyter nbconvert --TagRemovePreprocessor.enabled=True \
        --TagRemovePreprocessor.remove_cell_tags solution \
        --to notebook --output-dir student/ docs/source/tutorials/*.ipynb

The same course exists in Rust: see the ``guide`` module of the ``audio_samples``
crate (`docs.rs <https://docs.rs/audio_samples/latest/audio_samples/guide/>`_,
available from the next release; until then build the docs locally with
``cargo doc --features educational,full_no_plotting --open``) and its
``examples/tutorial_NN_*.rs`` labs.

.. toctree::
   :maxdepth: 1

   01_sampling_aliasing
   02_quantisation_dither
   03_dft_windows
   04_stft_spectrograms
   05_convolution_fir
   06_iir_filters
   07_resampling
   08_lpc_formants
   09_pitch
   10_cepstrum_mel_mfcc
   11_energy_vad
   12_dynamics
   13_onsets_tempo
   14_psychoacoustics_codecs
