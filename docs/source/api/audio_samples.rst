AudioSamples Core
================

.. currentmodule:: audio_samples

The AudioSamples class is the core of the library, providing a type-safe audio representation with
intrinsically embedded properties (sample rate, channel layout, format) that eliminates manual
metadata coordination.

AudioSamples Class
------------------

.. autoclass:: AudioSamples
   :members:
   :undoc-members:
   :show-inheritance:

   The main audio processing class supporting multiple sample formats and comprehensive audio operations.

   .. rubric:: Constructors

   .. autosummary::
      :nosignatures:

      ~AudioSamples.new_mono
      ~AudioSamples.new_multi

   .. rubric:: Factory Methods - Zeros

   .. autosummary::
      :nosignatures:

      ~AudioSamples.zeros_mono
      ~AudioSamples.zeros_mono_i16
      ~AudioSamples.zeros_mono_i32
      ~AudioSamples.zeros_mono_f64
      ~AudioSamples.zeros_multi
      ~AudioSamples.zeros_multi_i16
      ~AudioSamples.zeros_multi_i32
      ~AudioSamples.zeros_multi_f64

   .. rubric:: Factory Methods - Ones

   .. autosummary::
      :nosignatures:

      ~AudioSamples.ones_mono
      ~AudioSamples.ones_mono_i16
      ~AudioSamples.ones_mono_i32
      ~AudioSamples.ones_mono_f64
      ~AudioSamples.ones_multi
      ~AudioSamples.ones_multi_i16
      ~AudioSamples.ones_multi_i32
      ~AudioSamples.ones_multi_f64

   .. rubric:: Factory Methods - Uniform Values

   .. autosummary::
      :nosignatures:

      ~AudioSamples.uniform_mono
      ~AudioSamples.uniform_mono_i16
      ~AudioSamples.uniform_mono_i32
      ~AudioSamples.uniform_mono_f64
      ~AudioSamples.uniform_multi
      ~AudioSamples.uniform_multi_i16
      ~AudioSamples.uniform_multi_i32
      ~AudioSamples.uniform_multi_f64

   .. rubric:: Audio Editing

   .. autosummary::
      :nosignatures:

      ~AudioSamples.concatenate
      ~AudioSamples.stack
      ~AudioSamples.repeat
      ~AudioSamples.trim_silence
      ~AudioSamples.pad
      ~AudioSamples.split
      ~AudioSamples.mix
      ~AudioSamples.fade_in
      ~AudioSamples.fade_out

   .. rubric:: Channel Operations

   .. autosummary::
      :nosignatures:

      ~AudioSamples.pan
      ~AudioSamples.balance
      ~AudioSamples.to_mono
      ~AudioSamples.to_stereo
      ~AudioSamples.extract_channel
      ~AudioSamples.swap_channels

   .. rubric:: Spectral Analysis

   .. autosummary::
      :nosignatures:

      ~AudioSamples.stft
      ~AudioSamples.istft
      ~AudioSamples.spectrogram
      ~AudioSamples.mel_spectrogram
      ~AudioSamples.mfcc
      ~AudioSamples.chroma

   .. rubric:: Audio Processing

   .. autosummary::
      :nosignatures:

      ~AudioSamples.resample
      ~AudioSamples.resample_by_ratio
      ~AudioSamples.apply_window

   .. rubric:: Pitch Analysis

   .. autosummary::
      :nosignatures:

      ~AudioSamples.detect_pitch_yin
      ~AudioSamples.track_pitch

   .. rubric:: Audio Decomposition

   .. autosummary::
      :nosignatures:

      ~AudioSamples.hpss

   .. rubric:: Filtering

   .. autosummary::
      :nosignatures:

      ~AudioSamples.butterworth_lowpass
      ~AudioSamples.butterworth_highpass
      ~AudioSamples.butterworth_bandpass
      ~AudioSamples.low_pass_filter
      ~AudioSamples.high_pass_filter
      ~AudioSamples.band_pass_filter

   .. rubric:: Statistics and Analysis

   .. autosummary::
      :nosignatures:

      ~AudioSamples.peak
      ~AudioSamples.min_sample
      ~AudioSamples.max_sample
      ~AudioSamples.mean
      ~AudioSamples.rms
      ~AudioSamples.variance
      ~AudioSamples.std_dev
      ~AudioSamples.zero_crossings
      ~AudioSamples.zero_crossing_rate
      ~AudioSamples.autocorrelation
      ~AudioSamples.spectral_centroid
      ~AudioSamples.spectral_rolloff

   .. rubric:: Amplitude Processing

   .. autosummary::
      :nosignatures:

      ~AudioSamples.scale
      ~AudioSamples.normalize
      ~AudioSamples.clip
      ~AudioSamples.remove_dc_offset

   .. rubric:: Time-domain Processing

   .. autosummary::
      :nosignatures:

      ~AudioSamples.reverse
      ~AudioSamples.trim

   .. rubric:: Dynamic Range Processing

   .. autosummary::
      :nosignatures:

      ~AudioSamples.apply_compressor
      ~AudioSamples.apply_limiter
      ~AudioSamples.apply_gate
      ~AudioSamples.apply_expander

   .. rubric:: Equalization

   .. autosummary::
      :nosignatures:

      ~AudioSamples.apply_parametric_eq
      ~AudioSamples.apply_eq_band
      ~AudioSamples.apply_peak_filter
      ~AudioSamples.apply_low_shelf
      ~AudioSamples.apply_high_shelf
      ~AudioSamples.apply_three_band_eq

   .. rubric:: Frequency Analysis

   .. autosummary::
      :nosignatures:

      ~AudioSamples.frequency_response
      ~AudioSamples.fft
      ~AudioSamples.power_spectral_density

   .. rubric:: Properties and Metadata

   .. autosummary::
      :nosignatures:

      ~AudioSamples.dtype
      ~AudioSamples.sample_rate
      ~AudioSamples.num_channels
      ~AudioSamples.samples_per_channel
      ~AudioSamples.total_samples
      ~AudioSamples.shape
      ~AudioSamples.is_mono
      ~AudioSamples.is_multi_channel
      ~AudioSamples.is_empty
      ~AudioSamples.duration_seconds

Filter and EQ Configuration Classes
-----------------------------------

.. autoclass:: IirFilterDesign
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EqBand
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ParametricEq
   :members:
   :undoc-members:
   :show-inheritance: