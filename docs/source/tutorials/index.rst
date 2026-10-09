Tutorials: a short course in speech and audio processing
=========================================================

Fourteen self-checking notebooks, one per topic of a typical speech and audio
processing module. Each starts from a concept, shows it with ``audio_samples``,
re-derives the result from first principles with NumPy ("under the hood"), asks
the library to explain its own settings, and ends with exercises whose answers
are computed and asserted, so every number in the text is checked whenever the
notebook runs.

Run them locally with Jupyter, or check them all headlessly with::

    pytest --nbmake docs/source/tutorials

The same course exists in Rust: see the ``guide`` module of the ``audio_samples``
crate (`docs.rs <https://docs.rs/audio_samples/latest/audio_samples/guide/>`_) and
its ``examples/tutorial_NN_*.rs`` labs.

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
