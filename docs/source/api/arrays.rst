Array Helpers
=============

``audio_samples.arrays`` holds array helpers: local extrema, event and interval matching, one-dimensional interpolation, padding, reordering, compressed sparse rows, zero crossings, generated tones and sweeps, and argument checks.

.. automodule:: audio_samples.arrays
   :no-members:

.. currentmodule:: audio_samples.arrays

.. autodata:: MAX_MEM_BLOCK

.. autoclass:: SparseRows
   :members:

.. autofunction:: abs2

.. autofunction:: axis_sort

.. autofunction:: buf_to_float

.. autofunction:: chirp

.. autofunction:: count_unique

.. autofunction:: cyclic_gradient

.. autofunction:: dtype_c2r

.. autofunction:: dtype_r2c

.. autofunction:: expand_to

.. autofunction:: fill_off_diagonal

.. autofunction:: find_files

.. autofunction:: fix_frames

.. autofunction:: index_to_slice

.. autofunction:: interp1d

.. autofunction:: interp_broadcast

.. autofunction:: is_positive_int

.. autofunction:: is_unique

.. autofunction:: localmax

.. autofunction:: localmin

.. autofunction:: match_events

.. autofunction:: match_intervals

.. autofunction:: pad_center

.. autofunction:: phasor

.. autofunction:: shear

.. autofunction:: sparsify_rows

.. autofunction:: stack

.. autofunction:: tiny

.. autofunction:: tone

.. autofunction:: valid_audio

.. autofunction:: valid_int

.. autofunction:: valid_intervals

.. autofunction:: zero_crossings
