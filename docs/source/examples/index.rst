Examples
========

Where to find runnable code, by what you want to do.

.. toctree::
   :maxdepth: 1

   ../gallery/index

The :doc:`example gallery <../gallery/index>` has one short program per task, each with
its plot or audio player. The documentation build runs all of them.

The :doc:`tutorials <../tutorials/index>` are longer notebooks that explain the signal
processing behind each task and check every number they state.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Task
     - Gallery section
     - Tutorials
   * - Loading and saving
     - :ref:`Read and write <sphx_glr_gallery_loading_saving_plot_read_and_write.py>`,
       :ref:`read in blocks <sphx_glr_gallery_loading_saving_plot_stream_a_long_file.py>`
     - :doc:`Quantisation and dither <../tutorials/02_quantisation_dither>`
   * - Spectrograms
     - :ref:`STFT <sphx_glr_gallery_spectrograms_plot_stft_spectrogram.py>`,
       :ref:`constant-Q <sphx_glr_gallery_spectrograms_plot_constant_q.py>`
     - :doc:`DFT and windows <../tutorials/03_dft_windows>`,
       :doc:`STFT <../tutorials/04_stft_spectrograms>`
   * - Mel and MFCC
     - :ref:`Mel <sphx_glr_gallery_mel_mfcc_plot_mel_spectrogram.py>`,
       :ref:`MFCC <sphx_glr_gallery_mel_mfcc_plot_mfcc.py>`,
       :ref:`chroma <sphx_glr_gallery_mel_mfcc_plot_chroma.py>`,
       :ref:`perceptual scales <sphx_glr_gallery_mel_mfcc_plot_perceptual_scales.py>`
     - :doc:`Cepstrum, mel and MFCC <../tutorials/10_cepstrum_mel_mfcc>`,
       :doc:`Psychoacoustics and codecs <../tutorials/14_psychoacoustics_codecs>`
   * - Pitch
     - :ref:`YIN and pYIN <sphx_glr_gallery_pitch_plot_yin_and_pyin.py>`,
       :ref:`harmonics <sphx_glr_gallery_pitch_plot_harmonics.py>`
     - :doc:`Pitch <../tutorials/09_pitch>`, :doc:`LPC and formants <../tutorials/08_lpc_formants>`
   * - Beats and onsets
     - :ref:`Onsets <sphx_glr_gallery_beats_onsets_plot_onset_detection.py>`,
       :ref:`beats <sphx_glr_gallery_beats_onsets_plot_beat_tracking.py>`,
       :ref:`tempogram <sphx_glr_gallery_beats_onsets_plot_tempogram.py>`
     - :doc:`Onsets and tempo <../tutorials/13_onsets_tempo>`
   * - Separation
     - :ref:`Harmonic and percussive <sphx_glr_gallery_separation_plot_hpss.py>`
     -
   * - Effects
     - :ref:`Time stretch and pitch shift <sphx_glr_gallery_effects_plot_time_stretch_and_pitch_shift.py>`,
       :ref:`compressor and limiter <sphx_glr_gallery_effects_plot_compressor_and_limiter.py>`,
       :ref:`silence and speech <sphx_glr_gallery_effects_plot_silence_and_speech.py>`
     - :doc:`Dynamics <../tutorials/12_dynamics>`, :doc:`Energy and VAD <../tutorials/11_energy_vad>`
   * - Resampling
     - :ref:`Quality settings <sphx_glr_gallery_resampling_plot_resample_qualities.py>`,
       :ref:`batches with threads <sphx_glr_gallery_resampling_plot_batch_with_threads.py>`
     - :doc:`Sampling and aliasing <../tutorials/01_sampling_aliasing>`,
       :doc:`Resampling <../tutorials/07_resampling>`
   * - Filters
     - :ref:`Low-pass and high-pass <sphx_glr_gallery_filters_plot_lowpass_and_highpass.py>`,
       :ref:`parametric EQ <sphx_glr_gallery_filters_plot_parametric_eq.py>`
     - :doc:`FIR <../tutorials/05_convolution_fir>`, :doc:`IIR <../tutorials/06_iir_filters>`
   * - Plotting
     - :ref:`Waveform and spectrum <sphx_glr_gallery_plotting_plot_waveform_and_spectrum.py>`,
       :ref:`stereo <sphx_glr_gallery_plotting_plot_stereo_channels.py>`,
       :ref:`overview <sphx_glr_gallery_plotting_plot_overview_dashboard.py>`
     - Every tutorial plots its results

Going further
-------------

The Rust crate has the same tasks as examples: see the gallery chapter of its
`book <https://github.com/jmg049/audio_samples/tree/main/book>`_.
