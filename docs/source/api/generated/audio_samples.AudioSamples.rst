audio\_samples.AudioSamples
===========================

.. currentmodule:: audio_samples

.. autoclass:: AudioSamples

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~AudioSamples.__init__
      ~AudioSamples.add_numpy_array
      ~AudioSamples.apply_butterworth_bandpass
      ~AudioSamples.apply_butterworth_highpass
      ~AudioSamples.apply_butterworth_lowpass
      ~AudioSamples.apply_chebyshev_i
      ~AudioSamples.apply_compressor
      ~AudioSamples.apply_eq_band
      ~AudioSamples.apply_expander
      ~AudioSamples.apply_gate
      ~AudioSamples.apply_high_shelf
      ~AudioSamples.apply_iir_filter
      ~AudioSamples.apply_limiter
      ~AudioSamples.apply_low_shelf
      ~AudioSamples.apply_parametric_eq
      ~AudioSamples.apply_peak_filter
      ~AudioSamples.apply_three_band_eq
      ~AudioSamples.apply_window
      ~AudioSamples.as_f32
      ~AudioSamples.as_f64
      ~AudioSamples.as_i16
      ~AudioSamples.as_i32
      ~AudioSamples.autocorrelation
      ~AudioSamples.balance
      ~AudioSamples.band_pass_filter
      ~AudioSamples.butterworth_bandpass
      ~AudioSamples.butterworth_highpass
      ~AudioSamples.butterworth_lowpass
      ~AudioSamples.cast_as
      ~AudioSamples.cast_as_f32
      ~AudioSamples.cast_as_f64
      ~AudioSamples.cast_as_i16
      ~AudioSamples.cast_as_i32
      ~AudioSamples.chroma
      ~AudioSamples.clip
      ~AudioSamples.concatenate
      ~AudioSamples.cross_correlation
      ~AudioSamples.detect_pitch_yin
      ~AudioSamples.div_numpy_array
      ~AudioSamples.extract_channel
      ~AudioSamples.fade_in
      ~AudioSamples.fade_out
      ~AudioSamples.fft
      ~AudioSamples.frequency_response
      ~AudioSamples.from_array
      ~AudioSamples.from_numpy_array_with_metadata
      ~AudioSamples.high_pass_filter
      ~AudioSamples.hpss
      ~AudioSamples.info
      ~AudioSamples.is_empty
      ~AudioSamples.is_mono
      ~AudioSamples.is_multi_channel
      ~AudioSamples.istft
      ~AudioSamples.low_pass_filter
      ~AudioSamples.max
      ~AudioSamples.mean
      ~AudioSamples.mel_spectrogram
      ~AudioSamples.mfcc
      ~AudioSamples.min
      ~AudioSamples.mix
      ~AudioSamples.mul_numpy_array
      ~AudioSamples.new_mono
      ~AudioSamples.new_multi
      ~AudioSamples.normalize
      ~AudioSamples.ones_mono
      ~AudioSamples.ones_mono_f64
      ~AudioSamples.ones_mono_i16
      ~AudioSamples.ones_mono_i32
      ~AudioSamples.ones_multi
      ~AudioSamples.ones_multi_f64
      ~AudioSamples.ones_multi_i16
      ~AudioSamples.ones_multi_i32
      ~AudioSamples.pad
      ~AudioSamples.pan
      ~AudioSamples.peak
      ~AudioSamples.power_spectral_density
      ~AudioSamples.remove_dc_offset
      ~AudioSamples.repeat
      ~AudioSamples.resample
      ~AudioSamples.resample_by_ratio
      ~AudioSamples.reverse
      ~AudioSamples.reverse_in_place
      ~AudioSamples.rms
      ~AudioSamples.samples_per_channel
      ~AudioSamples.scale
      ~AudioSamples.spectral_centroid
      ~AudioSamples.spectral_rolloff
      ~AudioSamples.spectrogram
      ~AudioSamples.split
      ~AudioSamples.stack
      ~AudioSamples.std_dev
      ~AudioSamples.stft
      ~AudioSamples.stft_with_freqs
      ~AudioSamples.sub_numpy_array
      ~AudioSamples.swap_channels
      ~AudioSamples.to_format
      ~AudioSamples.to_mono
      ~AudioSamples.to_numpy
      ~AudioSamples.to_stereo
      ~AudioSamples.to_torch
      ~AudioSamples.track_pitch
      ~AudioSamples.trim
      ~AudioSamples.trim_silence
      ~AudioSamples.uniform_mono
      ~AudioSamples.uniform_mono_f64
      ~AudioSamples.uniform_mono_i16
      ~AudioSamples.uniform_mono_i32
      ~AudioSamples.uniform_multi
      ~AudioSamples.uniform_multi_f64
      ~AudioSamples.uniform_multi_i16
      ~AudioSamples.uniform_multi_i32
      ~AudioSamples.variance
      ~AudioSamples.zero_crossing_rate
      ~AudioSamples.zero_crossings
      ~AudioSamples.zeros_mono
      ~AudioSamples.zeros_mono_f64
      ~AudioSamples.zeros_mono_i16
      ~AudioSamples.zeros_mono_i32
      ~AudioSamples.zeros_multi
      ~AudioSamples.zeros_multi_f64
      ~AudioSamples.zeros_multi_i16
      ~AudioSamples.zeros_multi_i32
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~AudioSamples.channels
      ~AudioSamples.dtype
      ~AudioSamples.duration_milliseconds
      ~AudioSamples.duration_seconds
      ~AudioSamples.len
      ~AudioSamples.ndim
      ~AudioSamples.num_channels
      ~AudioSamples.sample_rate
      ~AudioSamples.shape
      ~AudioSamples.size
      ~AudioSamples.total_samples
   
   