"""Type stubs for ``audio_samples.spectrograms``.

``audio_samples.spectrograms`` is the Python API of the ``spectrograms`` crate,
compiled into the ``audio_samples`` extension (``StftParams``, ``WindowType``,
``CqtParams``, ``ChromaParams``, ...). Its classes mirror the standalone
``spectrograms`` package, a declared dependency whose stubs are re-exported
here, but they are distinct runtime objects: pass the classes from
``audio_samples.spectrograms`` to ``AudioSamples`` methods.
"""

from typing import ClassVar

from spectrograms import *  # noqa: F403  (re-exported for type checking)

class ChromaNorm:
    """Chroma vector normalisation for ``ChromaParams(norm=...)``.

    Registered by ``audio_samples`` when the bundled ``spectrograms`` build
    lacks it.
    """

    none: ClassVar[ChromaNorm]
    l1: ClassVar[ChromaNorm]
    l2: ClassVar[ChromaNorm]
    max: ClassVar[ChromaNorm]
