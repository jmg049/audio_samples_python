Sequence Alignment and Decoding
===============================

``audio_samples.sequence`` holds dynamic time warping, Viterbi decoding,
transition matrices, recurrence quantification, recurrence and
cross-similarity matrices, lag conversions and frame synchronisation
(``sync``). Feature matrices are ``(n_features, n_frames)``; probability
matrices ``(n_states, n_steps)``.

``sparse=True`` returns ``arrays.SparseRows``, integer outputs are ``int64``,
``sync`` with an empty index list aggregates the whole signal into one
segment, and ``rqa`` backtracking follows the move actually taken.

.. automodule:: audio_samples.sequence
   :no-members:

.. currentmodule:: audio_samples.sequence

Dynamic time warping
--------------------

.. autofunction:: dtw

.. autofunction:: dtw_backtracking

Hidden Markov model decoding
----------------------------

.. autofunction:: viterbi

.. autofunction:: viterbi_discriminative

.. autofunction:: viterbi_binary

Transition matrices
-------------------

.. autofunction:: transition_uniform

.. autofunction:: transition_loop

.. autofunction:: transition_cycle

.. autofunction:: transition_local

Recurrence and similarity
-------------------------

.. autofunction:: recurrence_matrix

.. autofunction:: cross_similarity

.. autofunction:: recurrence_to_lag

.. autofunction:: lag_to_recurrence

.. autofunction:: rqa

Aggregation
-----------

.. autofunction:: sync

Example
-------

.. code-block:: python

   import numpy as np
   from audio_samples import sequence

   X = np.random.rand(12, 40)          # e.g. chroma of a reference
   Y = np.random.rand(12, 55)          # chroma of a performance
   D, wp = sequence.dtw(X, Y, metric="cosine")
   # wp[::-1] runs from (0, 0) to (39, 54)

   trans = sequence.transition_loop(2, 0.9)
   voiced = sequence.viterbi_binary(np.random.rand(1, 100), trans)
