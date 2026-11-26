from . import arrays as arrays, effects as effects, explain as explain, factorization as factorization, features as features, inverse as inverse, io as io, linalg as linalg, notation as notation, pitch_filterbank as pitch_filterbank, rhythm as rhythm, segment as segment, sequence as sequence, spectral as spectral, types as types, utils as utils
from .explain import ExplainDoc
from .types import BeatTrackingData, Key, PitchContour, Psd, ResamplingQuality, SampleType
from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence
from numpy.typing import ArrayLike
from typing import Any, Literal, final

@final
class AudioSamples:
    """
    Represents homogeneous audio samples with associated metadata.

    Primary container for audio data combining raw sample values with essential
    metadata including sample rate, channel layout, and type information.
    Supports both mono and multi-channel audio with unified interface.

    The frame-wise analysis methods (``stft_matrix``, ``melspectrogram``,
    ``chroma_stft``, ``cqt``, ``onset_strength``, ``yin``, ...) return float32 /
    complex64 arrays for uint8, int16, int24 and float32 audio, and float64 /
    complex128 arrays for int32 and float64 audio: the smallest float type that
    holds every sample value exactly. Audio-returning methods keep the sample
    type.
    """
    def __abs__(self, /) -> AudioSamples:
        """
        `abs(audio)`, in this audio's sample type. Integer results saturate, so `abs(-32768)` is
        32767 on int16, and uint8 is measured from silence (128), so a stored 100 gives 156.
        """
    def __add__(self, other: object, /) -> AudioSamples: ...
    def __array__(self, /, dtype: Incomplete | None = None, copy: bool | None = None) -> Any:
        """
        NumPy array protocol. ``copy=True`` (NumPy 2) returns a fresh copy.
        """
    def __array_function__(self, /, func: Any, types: Any, args: tuple, kwargs: dict) -> Any:
        """
        NumPy array-function protocol (``np.mean(audio)``, ``np.fft.rfft(audio)``).

        ``AudioSamples`` arguments (also inside lists and tuples) are converted
        with :meth:`to_numpy` and the NumPy function is called on the arrays.
        The result is whatever NumPy returns (an array or a scalar).
        """
    @property
    def __array_interface__(self, /) -> dict: ...
    def __array_ufunc__(self, /, ufunc: Any, method: str, *inputs, **kwargs) -> Any:
        """
        NumPy ufunc protocol (``np.sin(audio)``, ``np.abs(audio)``, ``arr + audio``).

        Every ``AudioSamples`` input is converted with :meth:`to_numpy` and the
        ufunc is applied to the plain arrays, so ``np.sin(audio)`` returns the
        same NumPy array as ``np.sin(audio.to_numpy())``. The arithmetic ufuncs
        ``add``, ``subtract``, ``multiply`` and ``divide`` (which NumPy calls for
        ``ndarray + audio`` and friends) are the exception: when the result
        keeps this buffer's shape and a supported sample dtype it is wrapped
        back into an ``AudioSamples`` with this buffer's sample rate, matching
        ``audio + ndarray``.

        Raises:

            AudioParameterError: If two ``AudioSamples`` inputs have different
                sample rates (a ``ValueError``).
        """
    @classmethod
    def __class_getitem__(cls, /, item: Any) -> Any:
        """
        Returns the total number of samples for `len(audio)`.

        For mono audio, returns ``samples_per_channel``.
        For multi-channel audio, returns ``total_samples`` (channels x ``samples_per_channel``).

        Examples:

            >>> mono = aus.AudioSamples.from_array(np.zeros(44100), 44100)
            >>> len(mono)
            44100
            >>> stereo = aus.AudioSamples.from_array(np.zeros((2, 44100)), 44100)
            >>> len(stereo)
            88200
        `AudioSamples[numpy.float32]` names audio of that sample type in annotations; the type
        stubs use it to tie the precision of analysis results to the sample type.
        """
    def __eq__(self, other: object, /) -> bool:
        """
        Two `AudioSamples` are equal when they have the same sample type, sample
        rate, channel count and length, and every sample is equal. NaN samples
        compare unequal, as they do for floats.
        """
    def __getitem__(self, index: Any, /) -> Any: ...
    def __iadd__(self, other: object, /) -> None: ...
    def __imul__(self, other: object, /) -> None: ...
    def __isub__(self, other: object, /) -> None: ...
    def __itruediv__(self, other: object, /) -> None: ...
    def __len__(self, /) -> int: ...
    def __mul__(self, other: object, /) -> AudioSamples: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __neg__(self, /) -> AudioSamples:
        """
        `-audio`, in this audio's sample type. Integer results saturate, so `-(-32768)` is 32767 on
        int16, and uint8 mirrors about silence (128), so a stored 0 gives 255.
        """
    def __pos__(self, /) -> AudioSamples:
        """
        `+audio`: a copy of this audio.
        """
    def __pow__(self, exponent: object, modulo: object = None, /) -> AudioSamples: ...
    def __radd__(self, scalar: object, /) -> AudioSamples: ...
    def __repr__(self, /) -> str:
        """
        The :meth:`__str__` line followed by one line per channel (up to eight)
        previewing its first and last three samples.
        """
    def __rmul__(self, scalar: object, /) -> AudioSamples: ...
    def __rsub__(self, scalar: object, /) -> AudioSamples: ...
    def __rtruediv__(self, _scalar: object, /) -> AudioSamples: ...
    def __str__(self, /) -> str:
        """
        One line: sample type, channels, samples per channel, sample rate and
        duration, e.g. ``AudioSamples<f32>: 2 ch × 48000 samples @ 48000 Hz (1.000 s)``.
        """
    def __sub__(self, other: object, /) -> AudioSamples: ...
    def __truediv__(self, other: object, /) -> AudioSamples: ...
    @staticmethod
    def _new_mono_f32_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_mono_f64_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_mono_i16_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_mono_i24_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_mono_i32_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_multi_f32_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_multi_f64_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_multi_i16_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_multi_i24_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def _new_multi_i32_from_np(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples: ...
    def _repr_html_(self, /) -> str | None:
        """
        Rich HTML view for Jupyter: a summary table (dtype, channels, sample
        rate, duration, peak, RMS) and, for clips up to 30 s with one or two
        channels, an inline ``<audio>`` player (16-bit PCM WAV, base64).

        Never raises: on any failure it returns ``None`` and Jupyter falls back
        to the plain-text ``repr``.
        """
    def _resample_array(self, /, target_sample_rate: int, quality: ResamplingQuality | Literal["fast", "medium", "high", "very_high"], gain: float | None = None) -> Any:
        """
        ``resample`` returning the samples as an array of their own type. Float samples are
        moved into the array rather than copied out of an ``AudioSamples``.

        Args:

            ``target_sample_rate`` (int): As ``resample``.
            quality (ResamplingQuality | str): As ``resample``.
            gain (float | None): Float samples only: each output ``y`` becomes ``y * gain``
                computed in float64 and rounded to the sample type, as it is written.

        Returns:

            numpy.ndarray: Time on the last axis.
        """
    def _stft_matrix_into(self, /, out: Any, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> int:
        """
        ``stft_matrix`` of float audio written into ``out``, which needs no memory for the
        matrix.

        Args:

            out (numpy.ndarray): Of ``complex64`` for float32 audio and ``complex128`` for
                float64 audio, shape ``(1 + n_fft // 2, frames)`` for mono audio or
                ``(channels, 1 + n_fft // 2, frames)``, with at least as many frames as the
                result. Only the first frames that the result needs are written.
            ``n_fft``, ``hop_length``, ``win_length``, window, center, ``pad_mode``: As ``stft_matrix``.

        Returns:

            int: The number of frames written.
        """
    def add_numpy_array(self, /, numpy_array: "NDArray[Any]") -> AudioSamples: ...
    def amplitude(self, /) -> "int | float":
        """
        Return the peak (maximum absolute value) across all samples and channels.

        Alias for ``peak``, provided to match the conventional term "amplitude"
        used in some audio contexts.

        Returns:

            int | float: Maximum absolute amplitude present in the buffer.
        """
    def amplitude_envelope(self, /) -> "NDArray[Any]":
        """
        Compute a rectified amplitude envelope.

        Returns:

            numpy.ndarray: Envelope array shaped like the input (channels by samples).

        Raises:

            AudioError: If envelope computation fails.
        """
    def analytic_envelope(self, /) -> "NDArray[Any]":
        """
        Compute the analytic envelope via the Hilbert transform.

        Returns:

            numpy.ndarray: Instantaneous amplitude shaped like the input.

        Raises:

            AudioError: If envelope computation fails.
        """
    def apply_bessel_bandpass(self, /, order: "int", low_frequency: "float", high_frequency: "float") -> None:
        """
        Apply a Bessel band-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_bessel_bandstop(self, /, order: "int", low_frequency: "float", high_frequency: "float") -> None:
        """
        Apply a Bessel band-stop filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_bessel_highpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Bessel high-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): -3 dB cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_bessel_lowpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Bessel low-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Bessel filters have a maximally flat group delay in the passband,
        preserving the wave shape of in-band signals. The cutoff is the -3 dB point.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): -3 dB cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_butterworth_bandpass(self, /, order: "int", low_frequency: "float", high_frequency: "float") -> None:
        """
        Apply a Butterworth band-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            low_frequency (float): Lower cutoff frequency in hertz.
            high_frequency (float): Upper cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def apply_butterworth_highpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Butterworth high-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def apply_butterworth_lowpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Butterworth low-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def apply_chebyshev_i(self, /, order: "int", cutoff_frequency: "float", passband_ripple: "float", response: "FilterResponse | str") -> None:
        """
        Apply a Chebyshev Type I filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Cutoff frequency in hertz.
            passband_ripple (float): Maximum ripple within the passband in decibels.
            response (FilterResponse | str): ``'lowpass'`` or ``'highpass'``. For
                band designs use :meth:`IirFilterDesign.chebyshev_i_band` with
                :meth:`apply_iir_filter`.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def apply_chebyshev_ii_bandpass(self, /, order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float") -> None:
        """
        Apply a Chebyshev Type II band-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_chebyshev_ii_bandstop(self, /, order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float") -> None:
        """
        Apply a Chebyshev Type II band-stop filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_chebyshev_ii_highpass(self, /, order: "int", cutoff_frequency: "float", stopband_attenuation: "float") -> None:
        """
        Apply a Chebyshev Type II high-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Stopband edge frequency in hertz.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_chebyshev_ii_lowpass(self, /, order: "int", cutoff_frequency: "float", stopband_attenuation: "float") -> None:
        """
        Apply a Chebyshev Type II low-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Chebyshev Type II (inverse Chebyshev) filters have a maximally flat
        passband and equiripple stopband. They are specified by the stopband
        attenuation rather than passband ripple.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Stopband edge frequency in hertz.
            stopband_attenuation (float): Minimum stopband attenuation in decibels
                (typically 20-80 dB); must be positive.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_compressor(self, /, compressor_config: "CompressorConfig") -> None:
        """
        Apply a compressor configuration in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            compressor_config (CompressorConfig): Parameters controlling threshold, ratio, and timing.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If compression fails.
        """
    def apply_compressor_sidechain(self, /, config: "CompressorConfig", sidechain: "AudioSamples") -> AudioSamples:
        """
        Apply sidechain compression, returning a new buffer.

        The compressor's gain reduction is driven by the level of ``sidechain``
        rather than the main signal. Sidechain processing must be enabled on the
        configuration (set ``config.side_chain`` to an enabled ``SideChainConfig``).
        The detector input is ``(1 - m) * main + m * sidechain`` with
        ``m = config.side_chain.external_mix``, then the side-chain filters. A mono
        sidechain drives every channel; a multi-channel one must have as many
        channels as the main signal and drives them channel by channel.

        Note:

            Unlike the other dynamics operations in this module, which modify the
            signal in place, this returns a new buffer; both signals are copied
            first.

        Args:

            config (CompressorConfig): Compressor parameters with sidechain enabled.
            sidechain (AudioSamples): External control signal. Must match the main
                signal's dtype and length.

        Returns:

            AudioSamples: A new sidechain-compressed buffer.

        Raises:

            TypeError: If the main and sidechain dtypes differ.
            ValueError: If sidechain is not enabled, the configuration is invalid,
                the lengths differ, or a multi-channel sidechain has a different
                channel count.
            AudioError: If compression fails.
        """
    def apply_elliptic_bandpass(self, /, order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> None:
        """
        Apply an elliptic (Cauer) band-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.
            passband_ripple (float): Peak passband ripple in decibels; must be positive.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_elliptic_bandstop(self, /, order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> None:
        """
        Apply an elliptic (Cauer) band-stop filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Prototype filter order per edge; must be greater than zero.
            low_frequency (float): Lower band edge in hertz.
            high_frequency (float): Upper band edge in hertz.
            passband_ripple (float): Peak passband ripple in decibels; must be positive.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_elliptic_highpass(self, /, order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> None:
        """
        Apply an elliptic (Cauer) high-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Passband edge frequency in hertz.
            passband_ripple (float): Peak passband ripple in decibels; must be positive.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_elliptic_lowpass(self, /, order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> None:
        """
        Apply an elliptic (Cauer) low-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Elliptic filters are equiripple in both passband and stopband, giving the
        steepest transition for a given order. They are specified by both the
        passband ripple and the stopband attenuation.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Passband edge frequency in hertz.
            passband_ripple (float): Peak passband ripple in decibels; must be positive.
            stopband_attenuation (float): Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
    def apply_eq_band(self, /, band: "EqBand") -> None:
        """
        Apply a single EQ band in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            band (EqBand): Parametric band definition to apply.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_expander(self, /, config: "ExpanderConfig") -> None:
        """
        Apply a downward expander in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            config (ExpanderConfig): Expander parameters controlling threshold, ratio,
                and timing.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If expansion fails.
        """
    def apply_filter(self, /, coefficients: "np.typing.NDArray[np.floating]") -> AudioSamples:
        """
        Apply an FIR filter by "valid" linear convolution, returning a new buffer.

        For taps ``h[0..M]`` and a channel ``x`` of ``N`` samples this computes

        ``y[i] = sum_{k=0}^{M-1} h[k] * x[i + M - 1 - k]``,  ``i = 0 .. N - M``

        i.e. only the ``N - M + 1`` outputs where every tap overlaps the input.
        The result is ``numpy.convolve(x, h, mode='valid')`` applied to each
        channel independently. ``self`` is not modified.

        ``convolve`` vs ``apply_filter``: :meth:`convolve` returns the *full*
        convolution of length ``N + M - 1`` (``numpy.convolve(..., 'full')``) and
        takes another ``AudioSamples``; ``apply_filter`` takes a coefficient
        array and returns only the *valid* part of length ``N - M + 1``, so the
        output is shorter than the input by ``M - 1`` samples. Output index
        ``i`` corresponds to input time ``i + M - 1``; the filter's group delay
        (``(M - 1) / 2`` samples for a symmetric, linear-phase filter) is not
        compensated. Filters with 160 or more taps use FFT convolution; the
        result is the same up to rounding.

        Computation is carried out in float64 with float64 coefficients. Integer
        audio is filtered on its raw scale and the result is rounded to nearest
        and saturated to the sample type, so taps outside ``[-1, 1]`` work for
        every sample type.

        Args:

            coefficients (numpy.ndarray): 1-D array-like of FIR taps ``h``
                (non-empty, converted to float64). ``[1.0]`` leaves the signal
                unchanged.

        Returns:

            AudioSamples: Filtered buffer with ``N - M + 1`` samples per channel.

        Raises:

            ValueError: If ``coefficients`` is empty.
            AudioParameterError: If the audio is shorter than the filter.

        Examples:

            >>> import numpy as np, audio_samples as aus
            >>> x = aus.AudioSamples.new_mono(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), sample_rate=8000)
            >>> x.apply_filter(np.array([1.0, 0.5])).to_numpy()
            array([2.5, 4. , 5.5, 7. ])
        """
    def apply_filter_in_place(self, /, coefficients: "np.typing.NDArray[np.floating]") -> None:
        """
        Apply an FIR filter by "valid" convolution in place.

        Modifies the signal in place and returns ``None`` (the NumPy array it
        was created from, if any, is left unchanged).

        In-place twin of :meth:`apply_filter` (same formula and conventions):
        the buffer is replaced by the ``N - M + 1`` valid outputs, so it becomes
        ``M - 1`` samples shorter.

        Args:

            coefficients (numpy.ndarray): 1-D array-like of FIR taps (non-empty).

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If ``coefficients`` is empty.
            AudioParameterError: If the audio is shorter than the filter.
        """
    def apply_gate(self, /, config: "GateConfig") -> None:
        """
        Apply a downward noise gate in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            config (GateConfig): Gate parameters controlling threshold, ratio, and timing.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If gating fails.
        """
    def apply_high_shelf(self, /, frequency: "float", gain_db: "float", q_factor: "float") -> None:
        """
        Apply a high-shelf filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            frequency (float): Shelf corner frequency in hertz.
            gain_db (float): Gain adjustment in decibels.
            q_factor (float): Shelf slope parameter.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_iir_filter(self, /, design: "IirFilterDesign") -> None:
        """
        Apply an arbitrary IIR filter design in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            design (IirFilterDesign): Filter definition including coefficients and structure.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_limiter(self, /, config: "LimiterConfig") -> None:
        """
        Apply a limiter configuration in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            config (LimiterConfig): Parameters controlling ceiling, attack, and release.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If limiting fails.
        """
    def apply_limiter_sidechain(self, /, config: "LimiterConfig", sidechain: "AudioSamples") -> AudioSamples:
        """
        Apply sidechain limiting, returning a new buffer.

        The limiter's gain reduction ceiling is enforced based on the level of
        ``sidechain`` rather than the main signal. Sidechain processing must be
        enabled on the configuration. The detector input and the accepted channel
        combinations are as in :meth:`apply_compressor_sidechain`; with
        ``isp_limiting`` the detector reads true peaks.

        Note:

            Returns a new buffer rather than modifying the signal in place, as
            :meth:`apply_compressor_sidechain` does.

        Args:

            config (LimiterConfig): Limiter parameters with sidechain enabled.
            sidechain (AudioSamples): External control signal. Must match the main
                signal's dtype and length.

        Returns:

            AudioSamples: A new sidechain-limited buffer.

        Raises:

            TypeError: If the main and sidechain dtypes differ.
            ValueError: If sidechain is not enabled, the configuration is invalid,
                the lengths differ, or a multi-channel sidechain has a different
                channel count.
            AudioError: If limiting fails.
        """
    def apply_low_shelf(self, /, frequency: "float", gain_db: "float", q_factor: "float") -> None:
        """
        Apply a low-shelf filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            frequency (float): Shelf corner frequency in hertz.
            gain_db (float): Gain adjustment in decibels.
            q_factor (float): Shelf slope parameter.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_parametric_eq(self, /, eq: "ParametricEq") -> None:
        """
        Apply a parametric equalizer configuration in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            eq (ParametricEq): Equalizer definition containing one or more bands.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_peak_filter(self, /, frequency: "float", gain_db: "float", q_factor: "float") -> None:
        """
        Apply a parametric peak filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            frequency (float): Center frequency in hertz.
            gain_db (float): Gain adjustment in decibels.
            q_factor (float): Quality factor controlling bandwidth.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_three_band_eq(self, /, config: "ThreeBandEqConfig") -> None:
        """
        Apply a three-band equalizer in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            config (ThreeBandEqConfig): Three-band EQ configuration describing the
                low shelf, mid peak, and high shelf bands.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def apply_window(self, /, window: "np.typing.NDArray[np.floating]") -> AudioSamples:
        """
        Apply a sample-wise window to the signal.

        Each sample is multiplied by its float64 coefficient. Integer audio is
        multiplied on its raw scale and the products are rounded to nearest and
        saturated to the sample type, so fractional windows (Hann, Hamming, ...)
        work for every sample type.

        Args:

            window (numpy.ndarray): 1-D sequence of coefficients matching the
                audio length. Any real array-like is accepted (e.g. float32);
                it is converted to float64.

        Returns:

            AudioSamples: New buffer of the same sample type with the window
            applied.

        Raises:

            ValueError: If the window length does not equal the audio length.
            AudioError: If windowing fails.
        """
    def as_f32(self, /) -> AudioSamples:
        """
        Audio-aware conversion to float32; shorthand for ``to_format(SampleType.F32)``.

        Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
        (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
        Use ``cast_as_f32`` for a raw numeric cast instead.
        """
    def as_f64(self, /) -> AudioSamples:
        """
        Audio-aware conversion to float64; shorthand for ``to_format(SampleType.F64)``.

        Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
        (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
        Use ``cast_as_f64`` for a raw numeric cast instead.
        """
    def as_i16(self, /) -> AudioSamples:
        """
        Audio-aware conversion to int16; shorthand for ``to_format(SampleType.I16)``.

        Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
        (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
        Use ``cast_as_i16`` for a raw numeric cast instead.
        """
    def as_i32(self, /) -> AudioSamples:
        """
        Audio-aware conversion to int32; shorthand for ``to_format(SampleType.I32)``.

        Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
        (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
        Use ``cast_as_i32`` for a raw numeric cast instead.
        """
    def as_u8(self, /) -> AudioSamples:
        """
        Audio-aware conversion to uint8 (offset binary); shorthand for ``to_format(SampleType.U8)``.

        Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
        (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
        Use ``cast_as_u8`` for a raw numeric cast instead.
        """
    def attack_decay_envelope(self, /, follower: "EnvelopeFollower", method: "DynamicRangeMethod") -> tuple[Any, Any]:
        """
        Compute attack and decay envelopes using an envelope follower.

        Args:

            follower (EnvelopeFollower): Configuration describing attack and release behavior.
            method (DynamicRangeMethod): Detection strategy applied to the follower.

        Returns:

            tuple[numpy.ndarray, numpy.ndarray]: Attack and decay envelopes shaped like the input.

        Raises:

            AudioError: If envelope computation fails.
        """
    def autocorrelation(self, /, max_lag: int) -> list[float] | None:
        """
        Compute the autocorrelation sequence up to a lag limit.

        Uses the **unbiased** estimate
        ``r[k] = (1 / (N - k)) * sum_{n=0}^{N-1-k} x[n] * x[n + k]`` for
        lags ``0..=min(max_lag, N - 1)``, first channel only. It is not
        normalised: ``r[0]`` is the mean square, not 1. Divide by ``r[0]`` for a
        normalised sequence; multiply ``r[k]`` by ``(N - k) / N`` for the biased
        estimate. ``r[k] * (N - k)`` is the raw sum
        ``sum_{n=0}^{N-1-k} x[n] * x[n + k]``.

        Args:

            max_lag (int): Maximum lag to include; must be greater than zero.

        Returns:

            Optional[list[float]]: ``max_lag + 1`` values (fewer for short
                signals), or None if the FFT fails.

        Raises:

            ValueError: If max_lag is zero.
        """
    def balance(self, /, balance: "float") -> None:
        """
        Adjust the stereo balance in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            balance (float): Balance offset; negative favors left, positive favors right.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If the balance adjustment fails.
        """
    def band_pass_filter(self, /, low_cutoff_hz: "float", high_cutoff_hz: "float") -> AudioSamples:
        """
        Apply a band-pass filter (first-order Butterworth prototype), returning a new buffer.

        A second-order band-pass from the first-order Butterworth prototype by
        the bilinear transform, prewarped at both edges, run causally from a
        zero initial state. ``self`` is not modified.

        Args:

            low_cutoff_hz (float): Lower -3 dB edge in hertz.
            high_cutoff_hz (float): Upper -3 dB edge in hertz.

        Returns:

            AudioSamples: New buffer containing the filtered signal.

        Raises:

            AudioError: If filtering fails.
        """
    def beat_track(self, /, *, hop_length: int = ..., start_bpm: float = 120.0, tightness: float = 100.0, trim: bool = False, bpm: Any | None = None, prior: Any | None = None, units: str = "frames", sparse: bool = True) -> tuple[Any, Any]:
        """
        Tracks beats with the dynamic-programming beat tracker.

        The onset envelope is ``onset_strength(hop_length=hop_length,
        aggregate=numpy.median)``.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            start_bpm (float): Prior centre of the tempo estimate. Default 120.
            tightness (float): Tempo-consistency weight. Default 100.
            trim (bool): Remove weak leading/trailing beats. Default False.
            bpm (Optional[float | array_like]): Tempo to track at (one value or
                one per frame); None estimates it.
            prior: Tempo prior (see ``tempo``).
            units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
            sparse (bool): Must be True.

        Returns:

            tuple: ``(tempo, beats)``: ``tempo`` is the estimate as a
            float64 array of shape ``(1,)`` (``bpm`` unchanged when given, ``0.0``
            for silence); ``beats`` is a 1-D int64 (frames/samples) or float64
            (time) array.

        Raises:

            ValueError: For multi-channel audio, invalid parameters or
                ``sparse=False``.
        """
    def borrow_channel(self, /, channel_index: "int") -> AudioSamples:
        """
        Return a single channel as a new mono buffer.

        The underlying crate method ``borrow_channel`` returns a borrowed,
        lifetime-bound view into the buffer. Such views cannot be exposed to
        Python safely, so this binding copies the selected channel into an owned
        buffer before returning it.

        Args:

            channel_index (int): Zero-based index of the channel to borrow.
                Ignored for mono audio.

        Returns:

            AudioSamples: An owned mono buffer containing the selected channel.

        Raises:

            ValueError: If the channel index is out of range.
        """
    def butterworth_bandpass(self, /, order: "int", low_frequency: "float", high_frequency: "float") -> None:
        """
        Apply a Butterworth band-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            low_frequency (float): Lower cutoff frequency in hertz.
            high_frequency (float): Upper cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def butterworth_highpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Butterworth high-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def butterworth_lowpass(self, /, order: "int", cutoff_frequency: "float") -> None:
        """
        Apply a Butterworth low-pass filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            order (int): Filter order; must be greater than zero.
            cutoff_frequency (float): Cutoff frequency in hertz.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            ValueError: If order is zero.
            AudioError: If filter design or application fails.
        """
    def cast_as(self, /, dtype: SampleType) -> AudioSamples:
        """
        Cast to another sample type without audio-aware scaling.

        Raw numeric cast: each value keeps its number (like
        ``numpy.ndarray.astype`` with saturation), with no rescaling between
        full-scale ranges. Float-to-integer casts truncate toward zero and
        saturate at the type's limits, so float audio in ``[-1, 1]`` becomes
        mostly ``-1``, ``0`` and ``1``; int16 ``16384`` cast to float64 is
        ``16384.0``, not ``0.5``. Use :meth:`to_format` / ``as_*`` for
        audio-aware conversion.

        Args:

            dtype (SampleType): Target type, e.g. ``SampleType.F64``.

        Returns:

            AudioSamples: New buffer of the target type.

        Examples:

            >>> import numpy as np, audio_samples as aus
            >>> x = aus.AudioSamples.new_mono(np.array([16384, -32768], dtype=np.int16), sample_rate=8000)
            >>> x.cast_as(aus.SampleType.F64).to_numpy()
            array([ 16384., -32768.])
            >>> x.to_format(aus.SampleType.F64).to_numpy()
            array([ 0.50001526, -1.        ])
        """
    def cast_as_f32(self, /) -> AudioSamples:
        """
        Raw numeric cast to float32 (no rescaling); shorthand for
        ``cast_as(SampleType.F32)``. See :meth:`cast_as`; use ``as_f32``
        for audio-aware conversion.
        """
    def cast_as_f64(self, /) -> AudioSamples:
        """
        Raw numeric cast to float64 (no rescaling); shorthand for
        ``cast_as(SampleType.F64)``. See :meth:`cast_as`; use ``as_f64``
        for audio-aware conversion.
        """
    def cast_as_i16(self, /) -> AudioSamples:
        """
        Raw numeric cast to int16 (no rescaling); shorthand for
        ``cast_as(SampleType.I16)``. See :meth:`cast_as`; use ``as_i16``
        for audio-aware conversion.
        """
    def cast_as_i32(self, /) -> AudioSamples:
        """
        Raw numeric cast to int32 (no rescaling); shorthand for
        ``cast_as(SampleType.I32)``. See :meth:`cast_as`; use ``as_i32``
        for audio-aware conversion.
        """
    def cast_as_u8(self, /) -> AudioSamples:
        """
        Raw numeric cast to uint8 (no rescaling); shorthand for
        ``cast_as(SampleType.U8)``. See :meth:`cast_as`; use ``as_u8``
        for audio-aware conversion.
        """
    @property
    def channels(self, /) -> int:
        """
        Alias for ``num_channels``.
        """
    def chroma(self, /, stft_params: "StftParams", chroma_params: "ChromaParams") -> "NDArray[np.float64]":
        """
        Compute chromagram (pitch-class energy) features.

        Args:

            stft_params (StftParams): Parameters used to generate the STFT.
            chroma_params (ChromaParams): Configuration for chroma projection.

        Returns:

            numpy.ndarray: Chromagram shaped (pitch_classes, frames).

        Raises:

            AudioError: If chroma computation fails.
        """
    def chroma_cens(self, /, *, hop_length: int = ..., fmin: float | None = None, tuning: float | None = ..., n_chroma: int = ..., n_octaves: int = ..., bins_per_octave: int = ..., cqt_mode: str = "full", window: Any | None = None, norm: float | None = ..., win_len_smooth: int | None = ..., smoothing_window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
        """
        Chroma Energy Normalized Statistics (CENS) features.

        Args:

            hop_length, fmin, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode, window: As ``chroma_cqt``.
            norm (Optional[float]): Final per-frame normalisation. Default 2.
            win_len_smooth (Optional[int]): Smoothing length in frames (None
                disables). Default 41.
            smoothing_window: Smoothing window shape. Default ``"hann"``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def chroma_cqt(self, /, *, hop_length: int = ..., fmin: float | None = None, norm: float | None = ..., threshold: float | None = ..., tuning: float | None = ..., n_chroma: int = ..., n_octaves: int = ..., window: Any | None = None, bins_per_octave: int | None = ..., cqt_mode: str = "full") -> "NDArray[Any]":
        """
        Constant-Q chromagram.

        Args:

            hop_length (int): Default 512.
            fmin (Optional[float]): Lowest CQT frequency; None = C1.
            norm (Optional[float]): Per-frame normalisation. Default ``numpy.inf``.
            threshold (Optional[float]): Values below are zeroed. Default 0.0.
            tuning (Optional[float]): Tuning in fractions of a CQT bin. Default
                0.0; None estimates it from the signal.
            n_chroma (int): Default 12.
            n_octaves (int): Default 7.
            window: Must be None (a chroma smoothing window is not supported).
            bins_per_octave (Optional[int]): CQT resolution (multiple of
                ``n_chroma``). Default 36; None = ``n_chroma``.
            cqt_mode (str): ``"full"`` (default) or ``"hybrid"``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def chroma_stft(self, /, *, norm: float | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", tuning: float | None = ..., n_chroma: int = ..., ctroct: float = 5.0, octwidth: float | None = ..., base_c: bool = True) -> "NDArray[Any]":
        """
        Chromagram from the power STFT.

        ``chroma()`` (spectrograms crate) is a separate implementation; this one
        uses the ``chroma_filterbank`` with optional tuning estimation and
        per-frame ``norm`` normalisation.

        Args:

            norm (Optional[float]): Per-frame normalisation (``numpy.inf`` default,
                any p-norm, or None for raw energy).
            n_fft (int): FFT length. Default 2048.
            hop_length (int): Samples between frames. Default 512.
            win_length, window, center, pad_mode: STFT parameters (see ``stft_matrix``).
            tuning (Optional[float]): Tuning offset in fractions of a chroma bin.
                Default 0.0; None estimates it from the signal.
            n_chroma (int): Number of chroma bins. Default 12.
            ctroct (float): Centre of the octave weighting (octaves above A0). Default 5.
            octwidth (Optional[float]): Octave-weighting half-width; None = flat. Default 2.
            base_c (bool): Row 0 is C (True) or A. Default True.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``,
            also for mono audio (index ``[0]`` for ``(n_chroma, n_frames)``).

        Raises:

            ValueError: For invalid parameters.
        """
    def chroma_vqt(self, /, *, hop_length: int = ..., fmin: float | None = None, intervals: Any | None = None, norm: float | None = ..., threshold: float | None = ..., n_octaves: int = ..., bins_per_octave: int = ..., gamma: float | None = ...) -> "NDArray[Any]":
        """
        Variable-Q chromagram with equal-tempered bin spacing.

        Args:

            hop_length (int): Default 512.
            fmin (Optional[float]): Lowest VQT frequency; None = C1.
            intervals (str): Only ``"equal"`` (the default).
            norm (Optional[float]): Per-frame normalisation. Default ``numpy.inf``.
            threshold (Optional[float]): Values below are zeroed. Default 0.0.
            n_octaves (int): Default 7.
            bins_per_octave (int): VQT bins per octave (= chroma bins). Default 12.
            gamma (Optional[float]): Bandwidth offset (Hz); 0 (default) = CQT, None =
                ERB-derived.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, bins_per_octave, n_frames)``.

        Raises:

            ValueError: For invalid parameters or ``intervals`` other than ``"equal"``.
        """
    def clip(self, /, min_val: "float", max_val: "float") -> None:
        """
        Clamp samples in-place to the provided minimum and maximum.

        Modifies the signal in place and returns ``None`` (the NumPy array it
        was created from, if any, is left unchanged).

        Args:

            min_val (float): Lower bound for clamping.
            max_val (float): Upper bound for clamping.

        Raises:

            AudioError: If clamping fails or the bounds are invalid.
        """
    def complex_onset_detection(self, /, config: "ComplexOnsetConfig") -> list[float]:
        """
        Detect onset times using the complex-domain method.

        Args:

            config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.

        Returns:

            list[float]: Onset times in seconds.

        Raises:

            AudioError: If onset detection fails.
        """
    @classmethod
    def concatenate(cls, /, segments: "list[AudioSamples]") -> AudioSamples:
        """
        Concatenate multiple buffers end to end.

        Args:

            segments (list[AudioSamples]): Non-empty list of buffers to join.

        Returns:

            AudioSamples: Combined buffer spanning all segments.

        Raises:

            ValueError: If the segment list is empty.
            TypeError: If segments differ in dtype or are not AudioSamples.
            AudioError: If concatenation fails.
        """
    @classmethod
    def concatenate_owned(cls, /, segments: "list[AudioSamples]") -> AudioSamples:
        """
        Concatenate multiple owned buffers end to end.

        Behaves identically to :meth:`concatenate`; it mirrors the crate's
        owned-segment variant. All segments must share the same dtype, sample
        rate, and channel count.

        Args:

            segments (list[AudioSamples]): Non-empty list of buffers to join.

        Returns:

            AudioSamples: Combined buffer spanning all segments.

        Raises:

            ValueError: If the segment list is empty.
            TypeError: If segments differ in dtype or are not AudioSamples.
            AudioError: If concatenation fails.
        """
    def constant_q_transform(self, /, params: "CqtParams", hop_size: "int") -> "NDArray[np.complex128]":
        """
        Compute the Constant-Q Transform (CQT) of a mono signal.

        Args:

            params (CqtParams): CQT configuration (bins per octave, number of
                octaves, minimum frequency).
            hop_size (int): Hop size in samples between successive frames; must be
                greater than zero.

        Returns:

            numpy.ndarray: Complex CQT coefficients shaped (frequency_bins, time_frames).

        Raises:

            ValueError: If hop_size is zero.
            AudioError: If the signal is multi-channel or the computation fails.
        """
    def convolve(self, /, other: "AudioSamples") -> AudioSamples:
        """
        Compute the linear convolution of this signal with another.

        Both signals are converted to f64 internally. The result is mono and
        carries this signal's sample rate, with length
        ``len(self) + len(other) - 1``.

        This is the *full* convolution (``numpy.convolve(x, h, 'full')``). For
        FIR filtering with a coefficient array, :meth:`apply_filter` returns
        only the *valid* part, ``N - M + 1`` samples
        (``numpy.convolve(x, h, 'valid')``).

        Args:

            other (AudioSamples): Signal to convolve with. Must be mono and share
                this signal's data type and sample rate.

        Returns:

            AudioSamples: The convolved mono signal.

        Raises:

            AudioParameterError: If the two signals have different sample rates.
            TypeError: If the two signals have different data types.
            AudioError: If either signal is multi-channel or the FFT fails.
        """
    def copy(self, /) -> AudioSamples: ...
    def cqt(self, /, *, hop_length: int = ..., fmin: float | None = None, n_bins: int | None = ..., bins_per_octave: int = ..., tuning: float | None = ..., filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, pad_mode: str = "constant", res_type: str = "soxr_hq") -> "NDArray[Any]":
        """
        Constant-Q transform of every channel.

        ``constant_q_transform()`` (spectrograms crate) is a separate
        implementation with its own filters, scaling and framing.

        Args:

            hop_length (int): Samples between frames. Default 512.
            fmin (Optional[float]): Lowest bin frequency (Hz); None = C1 (32.70 Hz).
            n_bins (Optional[int]): Number of bins (default 84); None = as many as
                fit below Nyquist.
            bins_per_octave (int): Default 12.
            tuning (Optional[float]): Tuning offset in fractions of a bin (default
                0.0); None estimates it from the signal.
            filter_scale (float): Filter-length scale. Default 1.
            norm (Optional[float]): Filter normalisation (p-norm). Default 1.
            sparsity (float): Fraction of each filter's mass that may be dropped.
                Default 0.01.
            window: Named window of the filters. Default ``"hann"``.
            scale (bool): Divide by ``sqrt(filter length)``. Default True.
            pad_mode (str): Padding at the signal ends. Default ``"constant"``.
            res_type (str): Quality of the crate's resampler for the octave
                decimation: ``"soxr_hq"`` (default) high, ``"soxr_vhq"`` very
                high, ``"soxr_mq"`` medium, ``"soxr_lq"`` fast.

        Returns:

            numpy.ndarray[complex64 | complex128]: shape ``(channels, n_bins, n_frames)``,
            also for mono audio (index ``[0]`` for ``(n_bins, n_frames)``).

        Raises:

            ValueError: For invalid parameters (e.g. bins above Nyquist).
        """
    def cross_correlation(self, /, other: AudioSamples, max_lag: int) -> list[float]:
        """
        Compute the cross-correlation with another audio buffer.

        Uses the same unbiased, unnormalised estimate as :meth:`autocorrelation`:
        ``r[k] = (1 / (N - k)) * sum_n self[n] * other[n + k]`` for
        ``k = 0 .. max_lag``.

        Args:

            other (AudioSamples): Audio buffer to correlate against.
            max_lag (int): Maximum lag to include; must be greater than zero.

        Returns:

            list[float]: Cross-correlation coefficients up to the requested lag.

        Raises:

            ValueError: If max_lag is zero.
            AudioParameterError: If the sample rates or channel counts differ
                (the message names both values).
            TypeError: If the buffers have different data types.
        """
    def deconvolve(self, /, other: "AudioSamples", regularization: "float" = 0.0) -> AudioSamples:
        """
        Recover a system response by regularised spectral-division deconvolution.

        Solves ``self = denominator (*) h`` for ``h``. The ``regularization``
        term (>= 0) is scaled by the denominator's peak power to stabilise the
        division near spectral nulls. The result is mono at this signal's sample
        rate.

        Args:

            other (AudioSamples): The denominator signal. Must be mono and share
                this signal's data type.
            regularization (float): Non-negative regularisation factor.

        Returns:

            AudioSamples: The deconvolved mono signal.

        Raises:

            AudioParameterError: If the two signals have different sample rates.
            TypeError: If the two signals have different data types.
            AudioError: If either signal is multi-channel or the FFT fails.
        """
    def deemphasis(self, /, *, coef: float = 0.97) -> AudioSamples:
        """
        De-emphasis ``y[n] = x[n] + coef * y[n - 1]`` of every channel (inverse
        of ``preemphasis``).

        Args:

            coef (float): Coefficient (not 3). Default 0.97.

        Returns:

            AudioSamples: the filtered audio.

        Raises:

            ValueError: For a non-finite ``coef``, ``coef == 3`` or fewer than two
                samples.
        """
    def deemphasis_in_place(self, /, *, coef: float = 0.97) -> None:
        """
        In-place ``deemphasis``.
        """
    def deinterleave_channels(self, /) -> list[AudioSamples]:
        """
        Split a multi-channel buffer into individual mono buffers.

        Channel 0 becomes element 0, channel 1 becomes element 1, and so on. For
        mono input a single-element list is returned. This is the inverse of
        :meth:`interleave_channels`.

        Returns:

            list[AudioSamples]: One owned mono buffer per input channel.

        Raises:

            AudioError: If channel separation fails.
        """
    def detect_beats(self, /, config: "BeatTrackingConfig") -> BeatTrackingData:
        """
        Place beats on a fixed-tempo grid anchored at the strongest onset.

        Computes an onset strength envelope, then walks forward and backward
        from the global onset peak in steps of one inter-beat interval at
        ``config.tempo_bpm`` (snapping to nearby onsets within the configured
        tolerance). The tempo is *not* estimated: the returned ``tempo_bpm`` echoes
        the configuration. For tempo estimation and dynamic-programming beat
        tracking use ``beat_track``.

        Args:

            config (BeatTrackingConfig): Parameters controlling tempo targets and onset detection.

        Returns:

            BeatTrackingData: Result object containing estimated tempo, beat times, and metadata.

        Raises:

            AudioError: If beat tracking fails.
        """
    def detect_onsets(self, /, config: "OnsetDetectionConfig") -> list[float]:
        """
        Detect onset times using the configured algorithm.

        Args:

            config (OnsetDetectionConfig): Parameters controlling peak picking and preprocessing.

        Returns:

            list[float]: Onset times in seconds.

        Raises:

            AudioError: If onset detection fails.
        """
    def detect_onsets_spectral_flux(self, /, config: "SpectralFluxConfig") -> list[float]:
        """
        Detect onset times using spectral flux.

        Args:

            config (SpectralFluxConfig): Parameters configuring the flux computation and picker.

        Returns:

            list[float]: Onset times in seconds.

        Raises:

            AudioError: If onset detection fails.
        """
    def detect_pitch_autocorr(self, /, min_frequency: float, max_frequency: float) -> float | None:
        """
        Detect the fundamental frequency using autocorrelation.

        Args:

            min_frequency (float): Minimum expected frequency in hertz.
            max_frequency (float): Maximum expected frequency in hertz.

        Returns:

            float | None: Detected pitch in hertz, or None when no stable pitch is found.

        Raises:

            AudioError: If pitch detection fails.
        """
    def detect_pitch_yin(self, /, threshold: "float" = 0.1, min_frequency: "float" = 50.0, max_frequency: "float" = 2000.0) -> float | None:
        """
        Detect the fundamental frequency using the YIN algorithm.

        Args:

            threshold (float): Absolute threshold on YIN's cumulative mean
                normalised difference d'(tau), in [0, 1]. The first lag whose d'
                dips below it is taken. Lower values are **stricter** (a deeper
                dip is required, so fewer false detections and more ``None``);
                higher values are more permissive. 0.1-0.2 is typical.
            min_frequency (float): Minimum expected frequency in hertz.
            max_frequency (float): Maximum expected frequency in hertz.

        Returns:

            float | None: Detected pitch in hertz, or None when no stable pitch is found.

        Raises:

            AudioError: If pitch detection fails.
        """
    def div_numpy_array(self, /, numpy_array: "NDArray[Any]") -> AudioSamples: ...
    @property
    def dtype(self, /) -> str:
        """
        Returns the dtype as a string for Python property access
        """
    def duplicate_to_channels(self, /, n_channels: "int") -> AudioSamples:
        """
        Duplicate the buffer into the requested number of channels.

        Args:

            n_channels (int): Target channel count; must be greater than zero.

        Returns:

            AudioSamples: New buffer with the duplicated channels.

        Raises:

            AudioError: If duplication fails or the channel count is invalid.
        """
    @property
    def duration_milliseconds(self, /) -> float:
        """
        Duration of the audio in milliseconds (``duration_seconds * 1000``).
        """
    @property
    def duration_seconds(self, /) -> float:
        """
        Duration of the audio in seconds.

        Calculated as ``samples_per_channel`` / ``sample_rate``.

        Returns:

            float: Duration in seconds.

        Examples:

            >>> audio = aus.AudioSamples.from_array(np.zeros(44100), 44100)
            >>> audio.duration_seconds
            1.0
            >>> audio_3s = aus.AudioSamples.from_array(np.zeros(132300), 44100)
            >>> audio_3s.duration_seconds
            3.0
        """
    def eq_band(self, /, band: "EqBand") -> AudioSamples:
        """
        Apply a single EQ band, returning a new buffer.

        Non-mutating twin of :meth:`apply_eq_band`; leaves the current buffer
        unchanged.

        Args:

            band (EqBand): Parametric band definition to apply.

        Returns:

            AudioSamples: A new buffer with the band applied.

        Raises:

            AudioError: If filter application fails.
        """
    def eq_frequency_response(self, /, eq: "ParametricEq", frequencies: "np.typing.NDArray[np.float64]") -> tuple[Incomplete, Incomplete]:
        """
        Compute the combined magnitude and phase response of a parametric EQ.

        Evaluates each enabled band's biquad filter at every requested frequency
        and combines the results (magnitudes multiplied, phases summed), applying
        the EQ's output gain to the combined magnitude. Disabled bands are skipped.
        This is purely analytical and does not modify the audio.

        Args:

            eq (ParametricEq): The equalizer whose response to evaluate.
            frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz at
                which to evaluate the response.

        Returns:

            tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
            Linear magnitude (1.0 = unity gain) and phase (radians) arrays, each
            the same length as ``frequencies``.

        Raises:

            AudioError: If any enabled band fails to design a filter (e.g. a
                frequency above the Nyquist limit).
        """
    def estimate_key(self, /, stft_params: "StftParams") -> Key:
        """
        Estimate the musical key of the audio signal.

        Args:

            stft_params (StftParams): Parameters controlling the underlying STFT analysis.

        Returns:

            Key: Detected key, including tonic pitch class, mode (major/minor),
                and confidence score.

        Raises:

            AudioError: If key estimation fails.
        """
    def estimate_tempo(self, /, config: "BeatTrackingConfig") -> float:
        """
        Estimate the global tempo of the signal in beats per minute.

        This runs the same onset and tempo analysis as :meth:`detect_beats` but
        returns only the estimated tempo, skipping the per-beat tracking step.

        Args:

            config (BeatTrackingConfig): Parameters controlling tempo targets and
                onset detection.

        Returns:

            float: Estimated tempo in beats per minute.

        Raises:

            AudioError: If tempo estimation fails.
        """
    def estimate_tuning(self, /, *, n_fft: int = ..., resolution: float = 0.01, bins_per_octave: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1, ref: Any | None = None) -> float:
        """
        Tuning offset relative to A440 in fractions of a bin, estimated from
        spectral peaks.

        All channels are pooled.

        Args:

            n_fft (int): FFT length. Default 2048.
            resolution (float): Resolution in fractions of a bin. Default 0.01.
            bins_per_octave (int): Bins per octave of the grid. Default 12.
            hop_length, win_length, window, center, pad_mode, fmin, fmax, threshold, ref:
                ``piptrack`` parameters.

        Returns:

            float: tuning in ``[-0.5, 0.5)`` fractions of a bin.

        Raises:

            ValueError: For invalid parameters.
        """
    def explain(self, /, obj: "StftParams | SpectrogramParams | MelParams | MfccParams | WindowType | IirFilterDesign | EqBand | CompressorConfig | LimiterConfig | VadConfig | OnsetDetectionConfig", *, f0_range: "tuple[float, float] | None" = None, n_fft: "int | None" = None, n_mels: "int | None" = None) -> ExplainDoc:
        """
        Explain a configuration in the context of this signal.

        Uses this signal's sample rate, length (samples per channel) and
        channel count as context, so quantities such as frame counts, durations
        and "window shorter than a pitch period" warnings are computed for it.

        Args:

            obj: The configuration to explain (``StftParams``,
                ``SpectrogramParams``, ``MelParams``, ``MfccParams``,
                ``WindowType``, ``IirFilterDesign``, ``EqBand``,
                ``CompressorConfig``, ``LimiterConfig``, ``VadConfig`` or
                ``OnsetDetectionConfig``).
            f0_range (tuple[float, float] | None): Expected fundamental-frequency
                range ``(lowest, highest)`` in Hz.
            n_fft (int | None): FFT size a ``MelParams`` will be combined with.
            n_mels (int | None): Number of mel bands an ``MfccParams`` will be
                combined with.

        Returns:

            ExplainDoc: The explanation.

        Raises:

            TypeError: If ``obj`` is not one of the supported types.

        Examples:

            >>> import audio_samples as aus
            >>> audio = aus.sine_wave(440.0, 1.0, sample_rate=16000)
            >>> params = aus.StftParams(n_fft=512, hop_size=256, window=aus.WindowType.hanning)
            >>> doc = audio.explain(params)
            >>> doc.quantity("Frequency resolution")
            31.25
        """
    def extract_channel(self, /, channel_index: "int") -> AudioSamples:
        """
        Extract a single channel from the buffer.

        Args:

            channel_index (int): Zero-based index of the channel to extract.

        Returns:

            AudioSamples: Buffer representing the selected channel.

        Raises:

            AudioError: If the channel index is out of range or extraction fails.
        """
    def fade_in(self, /, duration_seconds: "float", curve: "FadeCurve") -> None:
        """
        Apply an in-place fade-in envelope.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            duration_seconds (float): Fade duration.
            curve (FadeCurve): Curve controlling the fade shape.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If the fade fails or parameters are invalid.
        """
    def fade_out(self, /, duration_seconds: "float", curve: "FadeCurve") -> None:
        """
        Apply an in-place fade-out envelope.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            duration_seconds (float): Fade duration.
            curve (FadeCurve): Curve controlling the fade shape.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If the fade fails or parameters are invalid.
        """
    def fft(self, /, n_fft: "int") -> "NDArray[np.complex128]":
        """
        Compute the Fast Fourier Transform (FFT) of the signal.

        The spectrum is **one-sided**: only the ``n_fft // 2 + 1`` non-negative
        frequency bins are returned, as ``numpy.fft.rfft(x, n_fft)`` per channel
        (the signal is zero-padded to ``n_fft``, which must be at least the
        signal length; longer input raises an error). Bin ``k`` is at
        ``k * sample_rate / n_fft`` Hz.

        Args:

            n_fft (int): FFT size in samples; must be greater than zero.

        Returns:

            numpy.ndarray: Complex spectrum shaped (channels, n_fft // 2 + 1).

        Raises:

            ValueError: If n_fft is zero.
            AudioError: If the FFT computation fails.
        """
    def filtfilt(self, /, design: "IirFilterDesign") -> None:
        """
        Apply a zero-phase (forward-backward) IIR filter in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Filters the signal twice -- once forward and once backward -- so the net
        phase response is zero. This doubles the effective filter order and
        squares the magnitude response. Useful when phase distortion must be
        avoided (e.g. analysis or offline processing).

        Args:

            design (IirFilterDesign): Filter definition to apply forward and backward.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If filter design or application fails.
        """
    def fourier_tempogram(self, /, *, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
        """
        Fourier tempogram of every channel.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            win_length (int): STFT length in frames. Default 384.
            center (bool): Centred frames. Default True.
            window: Analysis window. Default ``"hann"``.

        Returns:

            numpy.ndarray[complex64 | complex128]: shape
            ``(channels, win_length // 2 + 1, n_frames + 1)`` (centred); index
            ``[0]`` for the 2-D array of mono audio.
        """
    def frames(self, /) -> list:
        """
        Iterate over time-aligned frames of the signal.

        Each frame is a snapshot across all channels at a single time index, so
        the number of frames equals the number of samples per channel.

        Note:

            The crate yields each frame as a borrowed view; this method copies
            each frame into a new owned NumPy array.

        Returns:

            list[numpy.ndarray]: One array per time index, each shaped
                ``(1,)`` for mono audio or ``(channels, 1)`` for multi-channel
                audio.
        """
    def frequency_response(self, /, frequencies: "list[float]") -> tuple[list[float], list[float]]:
        """
        Deprecated: always returns a flat response.

        Audio samples do not carry a filter. Use
        :meth:`IirFilterDesign.frequency_response` or
        :meth:`SosFilter.frequency_response` instead. Emits a
        ``DeprecationWarning``.

        Args:

            frequencies (list[float]): Frequencies in hertz.

        Returns:

            tuple[list[float], list[float]]: All-ones magnitudes and all-zero phases.
        """
    @staticmethod
    def from_array(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples:
        """
        Create an ``AudioSamples`` instance from a NumPy array.

        Args:

            arr (numpy.ndarray): Audio data as a 1D (mono) or 2D (multi-channel) array.
                For multi-channel audio, shape should be (channels, samples).
                Supported dtypes: int16, int32, float32, float64.
            sample_rate (int): Sample rate in Hz (e.g., 44100, 48000).

        Returns:

            AudioSamples: A new audio samples object that wraps ``arr`` without
            copying it. In-place methods (``apply_*``, ``butterworth_*``,
            ``filtfilt``, ...) therefore modify ``arr`` itself; pass
            ``arr.copy()`` or call :meth:`copy` to keep the original. Read-only
            arrays are copied before the first in-place modification.

        Raises:

            TypeError: If array dtype is not supported or dimensions are invalid.

        Examples:

            >>> import numpy as np
            >>> import audio_samples as aus
            >>> # Create mono audio
            >>> mono = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
            >>> audio = aus.AudioSamples.from_array(mono, sample_rate=44100)
            >>> # Create stereo audio
            >>> stereo = np.random.randn(2, 44100).astype(np.float32)
            >>> audio = aus.AudioSamples.from_array(stereo, sample_rate=44100)
        """
    def from_numpy_array_with_metadata(self, /, numpy_array: "NDArray[Any]") -> AudioSamples: ...
    def gammatone_spectrogram(self, /, spec_params: "SpectrogramParams", erb_params: "ErbParams", log_params: "Optional[LogParams]" = None) -> "NDArray[np.float64]":
        """
        Compute a gammatone spectrogram using ERB-scaled filters.

        Args:

            spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
            erb_params (ErbParams): Configuration for the ERB filterbank.
            log_params (Optional[LogParams]): Optional parameters for decibel conversion.

        Returns:

            numpy.ndarray: Gammatone spectrogram shaped (bands, frames).

        Raises:

            AudioError: If spectrogram computation fails.

        Notes:

            ``log_params.floor_db`` is an absolute floor in dB re 1, not a
            range below the maximum (as ``top_db`` is).
        """
    def get_compression_curve(self, /, config: "CompressorConfig", input_levels_db: "np.typing.NDArray[np.float64]") -> "NDArray[np.float64]":
        """
        Compute the static compression input/output curve.

        Maps each input level through the compressor's static gain characteristic
        (threshold, ratio, knee) plus makeup gain. This is purely analytical and
        does not modify or use the audio content (no envelope following).

        Args:

            config (CompressorConfig): Compressor parameters.
            input_levels_db (np.typing.NDArray[np.float64]): Non-empty array of
                input levels in dBFS to evaluate.

        Returns:

            np.typing.NDArray[np.float64]: Output levels in dBFS, one per input,
            in the same order.

        Raises:

            ValueError: If ``input_levels_db`` is empty or the configuration is invalid.
            AudioError: If the computation fails.
        """
    def get_gain_reduction(self, /, config: "CompressorConfig") -> "NDArray[np.float64]":
        """
        Compute the per-sample gain reduction the compressor would apply.

        Passes the audio through the envelope follower and compression gain
        calculation, collecting the gain reduction (in dB) at every sample
        without modifying the signal. For multi-channel audio, only the first
        channel is analysed.

        Args:

            config (CompressorConfig): Compressor parameters.

        Returns:

            np.typing.NDArray[np.float64]: Gain reduction values in dB (each >= 0.0),
            one per sample (first channel for multi-channel audio).

        Raises:

            AudioError: If the configuration is invalid or processing fails.
        """
    @staticmethod
    def griffinlim_audio(S: "ArrayLike", *, sample_rate: int, n_iter: int = ..., hop_length: int | None = None, win_length: int | None = None, n_fft: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, length: int | None = None, pad_mode: str = "constant", momentum: float = 0.99, init: str | None = ..., rng: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = ..., dtype: SampleType | None = None) -> AudioSamples:
        """
        Reconstructs audio from a magnitude spectrogram with fast Griffin-Lim,
        returning ``AudioSamples``.

        This is a static method: ``AudioSamples.griffinlim_audio(S, sample_rate=sr)``.
        See ``audio_samples.effects.griffinlim`` for the arguments.

        Args:

            S (array_like): ``(1 + n_fft // 2, n_frames)`` or
                ``(channels, 1 + n_fft // 2, n_frames)`` non-negative magnitude.
            sample_rate (int): Sample rate of the result (keyword-only).
            **kwargs: ``n_iter``, ``hop_length``, ``win_length``, ``n_fft``,
                ``window``, ``center``, ``length``, ``pad_mode``, ``momentum``,
                ``init``, ``rng`` as ``audio_samples.effects.griffinlim``
                (``rng`` defaults to the seed 0, so repeated calls give the same
                audio).
            dtype (Optional[SampleType]): Sample type of the returned audio.
                ``None`` (default) gives float32 audio for a float32 ``S`` and
                float64 audio for any other input. The reconstruction runs in
                the float type of ``dtype`` (float32 for U8, I16, I24 and F32;
                float64 for I32 and F64); integer output is rounded and saturated.

        Returns:

            AudioSamples: audio of sample type ``dtype`` (mono for one channel).
        """
    @staticmethod
    def griffinlim_cqt(C: "ArrayLike", *, sample_rate: int, n_iter: int = ..., hop_length: int = ..., fmin: float | None = None, bins_per_octave: int = ..., tuning: float = 0.0, filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, pad_mode: str = "constant", res_type: str = "soxr_hq", length: int | None = None, momentum: float = 0.99, init: str | None = None, rng: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = ..., random_state: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = None, dtype: SampleType | None = None) -> AudioSamples:
        """
        Reconstructs audio from a constant-Q magnitude by Griffin-Lim.

        This is a static method. The default is the deterministic zero-phase
        start (``init=None``).

        Args:

            C (array_like): Non-negative CQT magnitude, ``(n_bins, T)`` or
                ``(channels, n_bins, T)``.
            sample_rate (int): Sample rate (keyword-only).
            n_iter (int): Iterations. Default 32.
            hop_length, fmin, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: The forward parameters (see ``cqt``).
            length (Optional[int]): Output length per channel.
            momentum (float): Fast Griffin-Lim momentum. Default 0.99.
            init (Optional[str]): None (default: zero phase, deterministic) or
                ``"random"``.
            rng: Seed for ``init="random"``, as ``griffinlim``: an integer (default 0),
                None for a fresh seed on every call, or a NumPy generator, from which
                one seed is drawn. The random phases come from this library's
                generator.
            random_state: Deprecated alias of ``rng``.
            dtype (Optional[SampleType]): Sample type of the returned audio.
                ``None`` (default) gives float32 audio for a float32 ``C`` and
                float64 audio for any other input. The computation runs in the
                float type of ``dtype`` (float32 for U8, I16, I24 and F32; float64
                for I32 and F64); integer output is rounded and saturated.

        Returns:

            AudioSamples: audio of sample type ``dtype``.

        Raises:

            ValueError: For invalid parameters or ``init`` other than None or
                ``"random"``.
        """
    def harmonic(self, /, *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> AudioSamples:
        """
        Harmonic component from median-filtering harmonic/percussive separation.

        STFT, ``audio_samples.effects.hpss`` (median filtering + soft masks),
        inverse STFT to the input length. The ``hpss`` method is separate and
        blends the masks its own way.

        Args:

            kernel_size (int | tuple[int, int]): Median filter lengths. Default 31.
            power (float): Mask exponent. Default 2.
            mask (bool): Must be False; the masks themselves are
                ``effects.hpss(S, mask=True)``.
            margin (float | tuple[float, float]): Margins (>= 1). Default 1.
            n_fft, hop_length, win_length, window, center, pad_mode: STFT
                parameters; ``window`` sets the STFT window.

        Returns:

            AudioSamples: the harmonic component, same length.
        """
    def harmonic_analysis(self, /, fundamental_freq: float, num_harmonics: int, tolerance: float, n_fft: int | None, window_type: Any | None) -> list[float]:
        """
        Measure harmonic amplitudes relative to a fundamental frequency.

        Args:

            fundamental_freq (float): Fundamental frequency in hertz.
            num_harmonics (int): Number of harmonics to report; must be greater than zero.
            tolerance (float): Frequency tolerance expressed as a fraction of the harmonic.
            n_fft (Optional[int]): Optional FFT size in samples; must be greater than zero when provided.
            window_type (Optional[WindowType]): Optional window type applied before the FFT.

        Returns:

            list[float]: Harmonic magnitudes ordered by harmonic index.

        Raises:

            ValueError: If num_harmonics or n_fft is zero.
            AudioError: If the analysis fails.
        """
    def harmonic_in_place(self, /, *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> None:
        """
        In-place ``harmonic``; arguments as ``harmonic``.
        """
    def harmonic_percussive(self, /, *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> tuple[AudioSamples, AudioSamples]:
        """
        Harmonic and percussive components from median-filtering separation
        (``hpss`` is a separate method with its own mask blend).

        Args:

            As ``harmonic``.

        Returns:

            tuple[AudioSamples, AudioSamples]: ``(harmonic, percussive)``; with
            margins of 1 they sum to the input up to round-off.
        """
    def harmonic_to_noise_ratio(self, /, fundamental_freq: float, num_harmonics: int, n_fft: int | None, window_type: Any | None) -> float:
        """
        Estimate the harmonic-to-noise ratio around a fundamental frequency.

        Args:

            fundamental_freq (float): Fundamental frequency in hertz.
            num_harmonics (int): Number of harmonics to evaluate; must be greater than zero.
            n_fft (Optional[int]): Optional FFT size in samples; must be greater than zero when provided.
            window_type (Optional[WindowType]): Optional window type applied before the FFT.

        Returns:

            float: Harmonic-to-noise ratio value.

        Raises:

            ValueError: If num_harmonics or n_fft is zero.
            AudioError: If the analysis fails.
        """
    def high_pass_filter(self, /, cutoff_hz: "float") -> AudioSamples:
        """
        Apply a first-order Butterworth high-pass filter, returning a new buffer.

        -3 dB at ``cutoff_hz``, 6 dB/octave: the bilinear-transform design
        prewarped at ``cutoff_hz``, run causally from a zero initial state.
        ``self`` is not modified.

        Args:

            cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).

        Returns:

            AudioSamples: New buffer containing the filtered signal.

        Raises:

            AudioError: If filtering fails.
        """
    def hpss(self, /, config: Any | None = None, median_filter_harmonic: int | None = None, median_filter_percussive: int | None = None, mask_softness: float | None = None, *, stft_params: Any | None = None) -> tuple[AudioSamples, AudioSamples]:
        """
        Separate harmonic and percussive components using median-filter HPSS.

        Args:

            config (StftParams | HpssConfig): Either a complete ``HpssConfig``
                (the remaining arguments must then be omitted), or the
                ``StftParams`` of the analysis STFT.
            median_filter_harmonic (Optional[int]): Harmonic (time-direction)
                median filter length in frames. Default 17. ``StftParams`` only.
            median_filter_percussive (Optional[int]): Percussive
                (frequency-direction) median filter length in bins. Default 17.
                ``StftParams`` only.
            mask_softness (Optional[float]): Mask softness in [0, 1] (0 = hard
                mask). Default 0.5. ``StftParams`` only.
            stft_params (Optional[StftParams]): Deprecated keyword alias of
                ``config`` (its name in audio_samples 0.2.x); emits a
                ``DeprecationWarning``.

        Returns:

            tuple[AudioSamples, AudioSamples]: ``(harmonic, percussive)``.

        Raises:

            TypeError: If ``config`` is neither ``StftParams`` nor ``HpssConfig``,
                or filter arguments are given together with an ``HpssConfig``.
            ValueError: If the configuration is invalid.

        Examples:

            >>> harmonic, percussive = audio.hpss(HpssConfig.musical)
            >>> harmonic, percussive = audio.hpss(StftParams(2048, 512, WindowType.hanning, True))
        """
    def hybrid_cqt(self, /, *, hop_length: int = ..., fmin: float | None = None, n_bins: int | None = ..., bins_per_octave: int = ..., tuning: float | None = ..., filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, pad_mode: str = "constant", res_type: str = "soxr_hq") -> "NDArray[Any]":
        """
        Hybrid constant-Q magnitude: pseudo-CQT for the high bins, full CQT for
        the low bins.

        Args:

            As ``cqt``.

        Returns:

            numpy.ndarray[float32 | float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def hybrid_tempogram(self, /, *, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
        """
        Hybrid tempogram of every channel: the Fourier and autocorrelation
        tempograms combined on the Fourier tempo grid.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            win_length (int): Window of both tempograms (>= 3). Default 384.
            center (bool): Centred frames. Default True.
            window: Window of both tempograms. Default ``"hann"``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, win_length // 2 + 1, n_frames)``
            on the ``fourier_tempo_frequencies`` grid; index ``[0]`` for the 2-D
            array of mono audio.
        """
    @staticmethod
    def icqt(C: "ArrayLike", *, sample_rate: int, hop_length: int = ..., fmin: float | None = None, bins_per_octave: int = ..., tuning: float = 0.0, filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, length: int | None = None, res_type: str = "soxr_hq", dtype: SampleType | None = None) -> AudioSamples:
        """
        Inverse constant-Q transform.

        This is a static method: ``AudioSamples.icqt(C, sample_rate=sr)``.

        Args:

            C (array_like): complex CQT of shape ``(n_bins, T)`` or ``(channels, n_bins, T)``.
            sample_rate (int): Sample rate of the analysed signal (keyword-only).
            hop_length, fmin, bins_per_octave, filter_scale, norm, sparsity, window, scale, res_type: The forward parameters (see ``cqt``).
            tuning (float): Tuning used by the forward transform. Default 0.0.
            length (Optional[int]): Exact output length per channel.
            dtype (Optional[SampleType]): Sample type of the returned audio.
                ``None`` (default) gives float32 audio for a complex64 ``C`` and
                float64 audio for any other input. The computation runs in the
                float type of ``dtype`` (float32 for U8, I16, I24 and F32; float64
                for I32 and F64); integer output is rounded and saturated.

        Returns:

            AudioSamples: audio of sample type ``dtype`` (mono for a single channel).

        Raises:

            ValueError: For invalid parameters or an empty ``C``.
        """
    def iirt(self, /, *, win_length: int = ..., hop_length: int | None = None, center: bool = True, tuning: float = 0.0, pad_mode: str = "constant", flayout: str = "sos", res_type: str = "soxr_hq", **kwargs) -> "NDArray[Any]":
        """
        The energy of each semitone band per frame.

        Args:

            ``win_length`` (int): Frame length in samples. Default 2048.
            ``hop_length`` (Optional[int]): Samples between frames. Default ``win_length // 4``.
            center (bool): Pad so that frames are centred. Default true.
            tuning (float): Deviation from A440 in fractions of a semitone.
            ``pad_mode`` (str): ``"constant"``, ``"edge"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``.
            flayout (str): ``"sos"`` (default) or ``"ba"``.
            ``res_type`` (str): Resampler quality.
            **kwargs: The filterbank's parameters.

        Returns:

            numpy.ndarray: ``(channels, 85, n_frames)``, ``float32`` for ``u8``, ``i16``,
            ``I24`` and ``f32`` audio, otherwise ``float64``.
        """
    def info(self, /) -> Any:
        """
        Get information about the audio samples.
        Returns:

            AudioSamplesInfo: An object containing sample rate, channels, frames, duration, and layout.

        Examples:

            >>> import audio_samples as aus
            >>> audio = aus.AudioSamples.from_array(np.random.randn(2, 44100).astype(np.float32), sample_rate=44100)
            >>> info = audio.info()
        """
    @staticmethod
    def interleave_channels(channels: "list[AudioSamples]") -> AudioSamples:
        """
        Combine multiple mono buffers into a single multi-channel buffer.

        The first buffer becomes channel 0, the second channel 1, and so on. All
        inputs must share the same dtype and number of samples; the output sample
        rate is taken from the first input.

        Args:

            channels (list[AudioSamples]): Non-empty list of mono buffers, all of
                the same dtype and length.

        Returns:

            AudioSamples: A multi-channel buffer with one channel per input.

        Raises:

            ValueError: If the list is empty, the dtypes differ, or the inputs do
                not all have the same sample count.
        """
    def is_mono(self, /) -> bool: ...
    def is_multi_channel(self, /) -> bool: ...
    @classmethod
    def istft(cls, /, stft_result: "StftResult") -> AudioSamples:
        """
        Invert a Short-Time Fourier Transform result.

        Args:

            stft_result (StftResult): Output from `stft` to be reconstructed.

        Returns:

            AudioSamples: Reconstructed time-domain signal.

        Raises:

            AudioError: If reconstruction fails.
        """
    @staticmethod
    def istft_matrix(stft_matrix: "ArrayLike", *, sample_rate: int, hop_length: int | None = None, win_length: int | None = None, n_fft: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, length: int | None = None, dtype: SampleType | None = None) -> AudioSamples:
        """
        Reconstruct audio from an STFT matrix by weighted overlap-add.

        This is a static method: ``AudioSamples.istft_matrix(D, sample_rate=sr)``.

        Args:

            stft_matrix (array_like): complex array of shape ``(channels, 1 + n_fft // 2, n_frames)``
                or ``(1 + n_fft // 2, n_frames)`` (treated as one channel).
            sample_rate (int): Sample rate of the returned audio (keyword-only).
            hop_length (Optional[int]): ``None`` means ``win_length // 4``.
            win_length (Optional[int]): ``None`` means ``n_fft``.
            n_fft (Optional[int]): ``None`` infers ``2 * (n_bins - 1)``.
            window: Window used for analysis (see ``stft_matrix``).
            center (bool): Whether the analysis frames were centred.
            length (Optional[int]): Exact output length per channel; pass the
                original length to undo the centre padding exactly.
            dtype (Optional[SampleType]): Sample type of the returned audio.
                ``None`` (default) gives float32 audio for a complex64 matrix and
                float64 audio for any other input. The inverse transform runs in
                the float type of ``dtype`` (float32 for U8, I16, I24 and F32;
                float64 for I32 and F64); integer output is rounded and saturated.

        Returns:

            AudioSamples: audio of sample type ``dtype``, mono for a single channel.

        Raises:

            ValueError: If the matrix is not 2-D/3-D, the bin count does not match
                ``n_fft``, or parameters are invalid.
        """
    def iter_channels(self, /) -> list:
        """
        Iterate over the complete signal of each channel.

        Each item is the full temporal sequence of one channel, yielded in
        increasing channel-index order. Named ``iter_channels`` to avoid clashing
        with :meth:`channels`, which returns the channel count.

        Note:

            Channel iteration copies each channel into a new owned NumPy array.

        Returns:

            list[numpy.ndarray]: One mono array per channel, each shaped
                ``(samples,)``.
        """
    @property
    def len(self, /) -> int:
        """
        Total number of samples across all channels (same as ``len(audio)``).
        """
    def low_pass_filter(self, /, cutoff_hz: "float") -> AudioSamples:
        """
        Apply a first-order Butterworth low-pass filter, returning a new buffer.

        -3 dB at ``cutoff_hz``, 6 dB/octave: the bilinear-transform design
        prewarped at ``cutoff_hz``, run causally from a zero initial state.
        ``self`` is not modified.

        Args:

            cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).

        Returns:

            AudioSamples: New buffer containing the filtered signal.

        Raises:

            AudioError: If filtering fails.
        """
    def lpc(self, /, *, order: int) -> "NDArray[Any]":
        """
        LPC coefficients of every channel (Burg's method).

        Args:

            order (int): Prediction order; each channel needs more than
                ``order`` samples.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, order + 1)``, rows
            ``[1, a_1, ..., a_order]``; mono audio gives one row (index ``[0]``
            for a 1-D result).

        Raises:

            ValueError: For a too-short or non-finite channel.
        """
    def magnitude_difference_matrix(self, /, config: "ComplexOnsetConfig") -> "NDArray[np.float64]":
        """
        Compute the complex-domain magnitude difference matrix.

        Args:

            config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.

        Returns:

            numpy.ndarray: Magnitude difference matrix with axes (frequency_bins, frames).

        Raises:

            AudioError: If matrix computation fails.
        """
    @staticmethod
    def magphase(complex_spect: "numpy.ndarray", power: "Optional[float]" = None) -> tuple[Any, Any]:
        """
        Decompose a complex spectrogram into magnitude and phase.

        Given a complex matrix ``D``, returns ``(magnitude, phase)`` such that
        ``D = magnitude * phase`` elementwise, where ``phase`` contains
        unit-magnitude complex factors. Bins where the magnitude is zero are
        assigned a phase of ``1 + 0j``.

        This is a static method that operates on a complex spectrogram array
        (such as the output of ``fft`` or ``constant_q_transform``) rather than on
        an ``AudioSamples`` instance.

        Args:

            complex_spect (numpy.ndarray): Complex 2-D STFT or FFT matrix.
            power (Optional[float]): Exponent applied to the magnitude values, as
                ``numpy.abs(D) ** power``; any number, fractional or negative.
                Defaults to 1 (raw magnitude).

        Returns:

            tuple[numpy.ndarray, numpy.ndarray]: The real-valued magnitude matrix
                and the complex unit-magnitude phase matrix.
        """
    def max(self, /) -> "int | float":
        """
        Compute the maximum sample value across all channels.

        Returns:

            int | float: Highest signed amplitude value found in the buffer.
        """
    def mean(self, /) -> float:
        """
        Calculate the arithmetic mean of the sample values.

        Returns:

            float: Mean amplitude across all samples and channels.
        """
    def median(self, /) -> float | None:
        """
        Return the value at the temporal midpoint, as ``midpoint_sample`` does.

        Samples are selected by index position; the buffer is not sorted.

        Returns:

            Optional[float]: The midpoint value if the view is non-empty, None otherwise.
        """
    def mel_spectrogram(self, /, spec_params: "SpectrogramParams", mel_params: "MelParams", log_params: "Optional[LogParams]" = None) -> "NDArray[np.float64]":
        """
        Compute a mel-frequency spectrogram.

        Args:

            spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
            mel_params (MelParams): Configuration for the mel filterbank.
            log_params (Optional[LogParams]): Optional parameters for decibel conversion.

        Returns:

            numpy.ndarray: Mel spectrogram shaped (mel_bins, frames). Without
                ``log_params`` the values are ``sqrt(M @ |X|**2)`` (the square root
                of the mel power spectrogram, following the ``spectrograms`` crate),
                which is neither the mel power ``M @ |X|**2`` nor the mel magnitude
                ``M @ |X|``. For either of those use ``melspectrogram()`` with
                ``power=2`` or ``power=1``.

        Raises:

            AudioError: If spectrogram computation fails.

        Notes:

            With ``M`` the mel filterbank and ``|X|**2`` the STFT power:

            - without ``log_params`` the result is ``sqrt(M @ |X|**2)``, the
              square root of the mel *power* (not ``M @ |X|``);
            - with ``log_params`` it is ``max(10*log10(M @ |X|**2), floor_db)``.
              ``floor_db`` is an **absolute** floor in dB re 1 (e.g. ``-80``
              clips values below -80 dB), not a range below the maximum like
              the ``top_db`` of ``power_to_db``.

            The mel scale is Slaney's (linear below 1 kHz, logarithmic above).
            The filters are **not** normalised by default
            (``MelParams(norm=None)``, triangles of peak 1); pass
            ``norm="slaney"`` for area normalisation.
        """
    def melspectrogram(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", power: float = 2.0, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
        """
        Mel spectrogram of every channel: the mel filterbank applied to the STFT
        magnitude raised to ``power``.

        Args:

            n_fft (int): Default 2048.
            hop_length (int): Default 512.
            win_length, window, center, pad_mode: As ``stft_matrix``.
            power (float): Exponent of the STFT magnitude; 2.0 (default) for power,
                1.0 for magnitude.
            n_mels (int): Number of mel bands. Default 128.
            fmin (float): Lowest frequency (Hz). Default 0.
            fmax (Optional[float]): Highest frequency (Hz); ``None`` = sr / 2.
            htk (bool): HTK mel scale instead of Slaney (default False).
            norm (None | "slaney" | float): Filter normalisation. Default "slaney".

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_mels, n_frames)``;
            float32 for uint8, int16, int24 and float32 audio, float64 for int32
            and float64 audio.

        Raises:

            ValueError: For invalid parameters.
        """
    def mfcc(self, /, stft_params: "StftParams", n_mels: "int", mfcc_params: "MfccParams") -> "NDArray[np.float64]":
        """
        Compute mel-frequency cepstral coefficients (MFCCs).

        Args:

            stft_params (StftParams): Parameters used to generate the STFT.
            n_mels (int): Number of mel bands; must be greater than zero.
            mfcc_params (MfccParams): Configuration for the cepstral projection.

        Returns:

            numpy.ndarray: MFCC matrix shaped (coefficients, frames). This follows
                the ``spectrograms`` crate's MFCC; ``mfcc_frames()`` is the
                configurable pipeline (orthonormal DCT, optional lifter and dB floor).

        Raises:

            ValueError: If n_mels is zero.
            AudioError: If the MFCC computation fails.

        Notes:

            Pipeline: mel power spectrogram (Slaney scale, ``n_mels`` bands from
            0 Hz to ``sample_rate / 2``, no filter normalisation), then
            ``L = 10*log10(max(., 1e-8))``, then an **unnormalised** DCT-II
            ``c[q] = sum_b L[b] * cos(pi*q*(b + 0.5)/n_mels)`` keeping the first
            ``n_mfcc`` coefficients, then the HTK sinusoidal lifter
            ``1 + (22/2)*sin(pi*q/22)`` (lifter 22).

            The result equals ``mfcc_frames`` (orthonormal DCT, no lifter, no dB
            floor) times ``s_q * lifter_q``, with ``s_0 = sqrt(n_mels)`` and
            ``s_q = sqrt(n_mels / 2)`` for ``q >= 1``.
        """
    def mfcc_frames(self, /, *, n_mfcc: int = ..., dct_type: int = ..., norm: str | None = ..., lifter: float = 0.0, mel_norm: Literal["slaney"] | float | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", power: float = 2.0, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, ref: float | Literal["max"] | Callable[..., Any] = ..., amin: float = 1e-10, top_db: float | None = None) -> "NDArray[Any]":
        """
        Mel-frequency cepstral coefficients of every channel.

        The pipeline is ``melspectrogram`` -> ``power_to_db(ref, amin, top_db)`` ->
        DCT along the mel axis -> first ``n_mfcc`` rows -> optional liftering.

        Args:

            n_mfcc (int): Number of coefficients. Default 20.
            dct_type (int): 1, 2 (default) or 3.
            norm (Optional[str]): DCT normalisation, ``"ortho"`` (default) or None.
            lifter (float): Liftering coefficient (0 disables). Default 0.
            mel_norm (None | "slaney" | float): Mel filter normalisation (``norm``
                of ``melspectrogram``). Default "slaney".
            n_fft, hop_length (default 512), win_length, window, center, pad_mode,
            power, n_mels, fmin, fmax, htk: Passed to the mel spectrogram, as in
                ``melspectrogram``.
            ref (float | "max" | numpy.max): dB reference. Default 1.0.
            amin (float): dB floor before the log. Default 1e-10.
            top_db (Optional[float]): Dynamic range floor in dB below the peak
                (per channel). Default None (no floor).

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_mfcc, n_frames)``;
            float32 for uint8, int16, int24 and float32 audio, float64 for int32
            and float64 audio.

        Raises:

            ValueError: For invalid parameters, including ``n_mfcc > n_mels``.
        """
    def midpoint_sample(self, /) -> float | None:
        """
        Return the value at the temporal midpoint of a mono signal.

        ``median`` returns the same value.
        For even-length signals the result is the average of the two central
        samples; for odd-length signals the single central sample is returned.
        Samples are selected by index position; the buffer is not sorted.

        Returns:

            Optional[float]: Midpoint value for mono audio, or None if the signal
                is multi-channel.
        """
    def min(self, /) -> "int | float":
        """
        Compute the minimum sample value across all channels.

        Returns:

            int | float: Lowest signed amplitude value found in the buffer.
        """
    @classmethod
    def mix(cls, /, sources: "list[AudioSamples]", weights: "Optional[list[float]]" = None) -> AudioSamples:
        """
        Mix multiple buffers together with optional weights.

        Args:

            sources (list[AudioSamples]): Non-empty list of buffers to mix.
            weights (Optional[list[float]]): Optional mixing weights, matching the number of sources.

        Returns:

            AudioSamples: Mixed buffer containing the weighted sum.

        Raises:

            ValueError: If the source list is empty.
            TypeError: If sources differ in dtype or are not AudioSamples.
            AudioError: If mixing fails.
        """
    def moving_average_envelope(self, /, window_size: "int", hop_size: "int") -> "NDArray[Any]":
        """
        Compute a moving-average envelope over the rectified signal.

        Args:

            window_size (int): Number of samples per averaging window; must be greater than zero.
            hop_size (int): Step between windows in samples; must be greater than zero.

        Returns:

            numpy.ndarray: Moving-average envelope shaped (channels, frames).

        Raises:

            ValueError: If window_size or hop_size is zero.
            AudioError: If envelope computation fails.
        """
    def mu_compress(self, /, mu: "float") -> AudioSamples:
        """
        Apply mu-law compression to the signal.

        Args:

            mu (float): Mu parameter controlling the compression curve.

        Returns:

            AudioSamples: New buffer containing the compressed samples.

        Raises:

            AudioError: If compression fails.
        """
    def mu_expand(self, /, mu: "float") -> AudioSamples:
        """
        Apply mu-law expansion to the signal.

        Args:

            mu (float): Mu parameter used during compression.

        Returns:

            AudioSamples: New buffer containing the expanded samples.

        Raises:

            AudioError: If expansion fails.
        """
    def mul_numpy_array(self, /, numpy_array: "NDArray[Any]") -> AudioSamples: ...
    @property
    def ndim(self, /) -> int:
        """
        Number of array dimensions: 1 for mono, 2 for multi-channel.
        """
    @staticmethod
    def new_mono(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples:
        """
        Create mono audio from a 1-D NumPy array.

        The array's dtype becomes the sample type; no value conversion is
        performed (float samples are expected in ``[-1, 1]``, integer samples
        span the full integer range). Supported dtypes: uint8, int16, int32,
        float32, float64.

        The returned object wraps ``arr`` without copying it (zero-copy). Methods
        that work in place (``apply_compressor``, ``apply_limiter``,
        ``apply_iir_filter``, ``filtfilt``, ``reverse_in_place``, ...) therefore
        modify ``arr`` itself; pass ``arr.copy()`` to keep the original.
        Methods that return a new buffer, or that replace the buffer wholesale
        (e.g. ``scale``), leave ``arr`` unchanged. Read-only arrays are copied
        before the first in-place modification.

        Args:

            arr (numpy.ndarray): 1-D array of samples (non-empty).
            sample_rate (int): Sample rate in Hz.

        Returns:

            AudioSamples: Mono audio viewing ``arr``.

        Raises:

            TypeError: If ``arr`` is not 1-D or its dtype is unsupported.
            ValueError: If ``arr`` is empty.

        Examples:

            >>> import numpy as np, audio_samples as aus
            >>> a = np.zeros(4)
            >>> x = aus.AudioSamples.new_mono(a, sample_rate=8000)
            >>> x.num_channels, x.sample_rate
            (1, 8000)
        """
    @staticmethod
    def new_multi(arr: "numpy.typing.ArrayLike", sample_rate: "int") -> AudioSamples:
        """
        Create multi-channel audio from a 2-D ``(channels, samples)`` NumPy array.

        Same conventions as :meth:`new_mono`: the dtype becomes the sample type
        and no value conversion is performed. A ``(samples, channels)``-shaped
        array triggers a warning.

        The returned object wraps ``arr`` without copying it (zero-copy). Methods
        that work in place (``apply_compressor``, ``apply_limiter``,
        ``apply_iir_filter``, ``filtfilt``, ``reverse_in_place``, ...) therefore
        modify ``arr`` itself; pass ``arr.copy()`` to keep the original.
        Methods that return a new buffer, or that replace the buffer wholesale
        (e.g. ``scale``), leave ``arr`` unchanged. Read-only arrays are copied
        before the first in-place modification.

        Args:

            arr (numpy.ndarray): 2-D array shaped ``(channels, samples)``.
            sample_rate (int): Sample rate in Hz.

        Returns:

            AudioSamples: Multi-channel audio viewing ``arr``.

        Raises:

            TypeError: If ``arr`` is not 2-D or its dtype is unsupported.
            ValueError: If ``arr`` is empty.

        Examples:

            >>> import numpy as np, audio_samples as aus
            >>> x = aus.AudioSamples.new_multi(np.zeros((2, 4), dtype=np.float32), sample_rate=8000)
            >>> x.num_channels
            2
        """
    def normalize(self, /, min: "float", max: "float", method: "NormalizationMethod | str") -> None:
        """
        Normalize samples in-place using one of the supported strategies.

        Modifies the signal in place and returns ``None`` (the NumPy array it
        was created from, if any, is left unchanged).

        Args:

            min (float): Lower bound for the normalized output.
            max (float): Upper bound for the normalized output.
            method (NormalizationMethod | str): Strategy controlling the normalization.

        Raises:

            AudioError: If normalization fails.
        """
    @property
    def num_channels(self, /) -> int:
        """
        Number of audio channels.

        Returns:

            int: Number of channels (1 for mono, 2 for stereo, etc.).

        Examples:

            >>> mono_audio.num_channels
            1
            >>> stereo_audio.num_channels
            2
        """
    def nyquist(self, /) -> float:
        """
        Nyquist frequency in Hz (``sample_rate / 2``).
        """
    @staticmethod
    def ones_mono(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_mono_f64(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_mono_i16(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_mono_i32(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_multi(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_multi_f64(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_multi_i16(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def ones_multi_i32(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    def onset_detect(self, /, *, hop_length: int = ..., backtrack: bool = False, energy: Any | None = None, units: str = "frames", normalize: bool = True, sparse: bool = True, pre_max: int | None = None, post_max: int | None = None, pre_avg: int | None = None, post_avg: int | None = None, delta: float | None = None, wait: int | None = None, method: str = "greedy") -> "NDArray[Any]":
        """
        Detects onsets by peak picking on the onset strength envelope.

        The default onset envelope (``onset_strength(hop_length=hop_length)``) is
        normalised to ``[0, 1]``, peak-picked and optionally backtracked.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            backtrack (bool): Move onsets back to the preceding local minimum of
                ``energy``. Default False.
            energy (Optional[array_like]): 1-D function to backtrack on (one value
                per frame); None uses the normalised envelope.
            units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
            normalize (bool): Scale the envelope to ``[0, 1]``. Default True.
            sparse (bool): Must be True.
            pre_max, post_max, pre_avg, post_avg, wait (Optional[int]),
            delta (Optional[float]), method (str): Peak-picking parameters
                (see ``rhythm.peak_pick``); None derives the value from the frame
                rate (30 ms / 100 ms windows, ``delta=0.07``).

        Returns:

            numpy.ndarray: 1-D onset frames or samples (int64), or times (float64).

        Raises:

            ValueError: For multi-channel audio, invalid parameters or
                ``sparse=False``.
        """
    def onset_detection_function(self, /, config: "OnsetDetectionConfig") -> tuple[list[float], list[float]]:
        """
        Compute the energy-based onset detection function.

        Args:

            config (OnsetDetectionConfig): Parameters controlling the detection function.

        Returns:

            tuple[list[float], list[float]]: ``(times, values)``: frame times in
                seconds first, then the detection-function value of each frame.

        Raises:

            AudioError: If the detection function computation fails.
        """
    def onset_detection_function_complex(self, /, config: "ComplexOnsetConfig") -> list[float]:
        """
        Compute the complex-domain onset detection function without peak picking.

        Args:

            config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.

        Returns:

            list[float]: Complex onset detection function values.

        Raises:

            AudioError: If detection function computation fails.
        """
    def onset_strength(self, /, *, lag: int = ..., max_size: int = ..., detrend: bool = False, center: bool = True, aggregate: Any | None = None, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., pad_mode: str = "constant", power: float = 2.0, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
        """
        Onset strength envelope of every channel.

        The envelope is the aggregated positive increase of the log-power mel
        spectrogram (``power_to_db`` with ``ref=1``, ``top_db=80``) from frame
        ``t - lag`` to ``t``.

        Args:

            lag (int): Time lag of the difference in frames. Default 1.
            max_size (int): Maximum filter across frequency on the reference
                (SuperFlux); 1 disables it. Default 1.
            detrend (bool): High-pass filter the envelope. Default False.
            center (bool): Compensate the centred-STFT delay. Default True.
            aggregate: Reduction across mel bins: None (mean), ``numpy.mean``,
                ``numpy.median``, ``numpy.max``, ``numpy.min``, ``numpy.sum``, a
                name, or a callable ``f(x, axis=-1)``.
            n_fft (int): Default 2048.
            hop_length (int): Default 512.
            win_length, window, pad_mode, power, n_mels, fmin, fmax, htk, norm:
                Mel spectrogram parameters, as ``melspectrogram``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_frames)`` with
            ``n_frames = 1 + len // hop_length``; index ``[0]`` for the
            ``(n_frames,)`` envelope of mono audio.

        Raises:

            ValueError: For invalid parameters.
        """
    def onset_strength_multi(self, /, *, channels: Any | None = None, lag: int = ..., max_size: int = ..., detrend: bool = False, center: bool = True, aggregate: Any | None = None, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., pad_mode: str = "constant", power: float = 2.0, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
        """
        Multi-band onset strength of every channel: one envelope per group of
        mel bands.

        Args:

            channels (Optional[Sequence[int] | Sequence[slice]]): Sub-band boundaries in
                mel bins: values above ``n_mels`` are dropped, the rest sorted and
                de-duplicated, each consecutive pair is one band. Or one slice of the
                mel bins per band, in any order. None gives one band.
            aggregate: As ``onset_strength``; ``False`` keeps every mel bin.
            **kwargs: ``lag``, ``max_size``, ``detrend``, ``center``, ``n_fft``,
                ``hop_length``, ``win_length``, ``window``, ``pad_mode``, ``power``,
                ``n_mels``, ``fmin``, ``fmax``, ``htk``, ``norm`` as ``onset_strength``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_bands, n_frames)``
            (``n_mels`` rows with ``aggregate=False``); index ``[0]`` for the
            ``(n_bands, n_frames)`` array of mono audio.

        Raises:

            ValueError: For invalid parameters or boundaries.
        """
    def pad(self, /, pad_start_seconds: "float", pad_end_seconds: "float", pad_value: "float") -> AudioSamples:
        """
        Pad the buffer with constant values on both sides.

        Args:

            pad_start_seconds (float): Duration to pad before the signal.
            pad_end_seconds (float): Duration to pad after the signal.
            pad_value (float): Value used for padding.

        Returns:

            AudioSamples: New buffer with the added padding.

        Raises:

            AudioError: If padding fails or durations are invalid.
        """
    def pad_samples_right(self, /, target_num_samples: "int", pad_value: "float") -> AudioSamples:
        """
        Right-pad the buffer to a fixed sample count.

        Args:

            target_num_samples (int): Desired total number of samples.
            pad_value (float): Value used for padding.

        Returns:

            AudioSamples: New buffer of ``target_num_samples`` samples. If the
                buffer is already at least that long it is returned unchanged
                (not truncated, no error).

        Raises:

            AudioError: If padding fails.
        """
    def pad_to_duration(self, /, target_duration_seconds: "float", pad_value: "float", pad_side: "PadSide") -> AudioSamples:
        """
        Pad the buffer to reach a target duration.

        Args:

            target_duration_seconds (float): Desired total duration.
            pad_value (float): Value used for padding.
            pad_side (PadSide): Side to apply the padding.

        Returns:

            AudioSamples: New buffer matching the requested duration.

        Raises:

            AudioError: If padding fails or the target duration is invalid.
        """
    def pan(self, /, pan_value: "float") -> None:
        """
        Apply stereo panning to the buffer in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            pan_value (float): Normalized pan value; negative favors left, positive favors right.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If the pan adjustment fails.
        """
    def parametric_eq(self, /, eq: "ParametricEq") -> AudioSamples:
        """
        Apply a parametric equalizer configuration, returning a new buffer.

        Non-mutating twin of :meth:`apply_parametric_eq`; leaves the current
        buffer unchanged.

        Args:

            eq (ParametricEq): Equalizer definition containing one or more bands.

        Returns:

            AudioSamples: A new equalized buffer.

        Raises:

            AudioError: If filter application fails.
        """
    def peak(self, /) -> "int | float":
        """
        Compute the peak sample value across all channels.

        Returns:

            int | float: Maximum absolute amplitude present in the buffer.
        """
    def percussive(self, /, *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> AudioSamples:
        """
        Percussive component from median-filtering harmonic/percussive separation.

        Args:

            As ``harmonic``.

        Returns:

            AudioSamples: the percussive component, same length.
        """
    def percussive_in_place(self, /, *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> None:
        """
        In-place ``percussive``; arguments as ``harmonic``.
        """
    def perturb(self, /, config: "PerturbationConfig") -> AudioSamples:
        """
        Apply stochastic perturbations and return a new buffer.

        Args:

            config (PerturbationConfig): Configuration describing perturbations to apply.

        Returns:

            AudioSamples: Perturbed buffer.

        Raises:

            AudioError: If perturbation fails.
        """
    def perturb_in_place(self, /, config: "PerturbationConfig") -> None:
        """
        Apply stochastic perturbations in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            config (PerturbationConfig): Configuration describing perturbations to apply.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If perturbation fails.
        """
    def phase_deviation_matrix(self, /, config: "ComplexOnsetConfig") -> "NDArray[np.float64]":
        """
        Compute the magnitude-weighted phase deviation matrix.

        Entry ``[b, t]`` is ``|X[b,t]| * |princarg(phi[b,t] - 2 phi[b,t-1] + phi[b,t-2])|``,
        the phase term of the complex-domain onset detection function: zero for
        any steady tone and for bins without signal. Columns 0 and 1 are zero.

        Args:

            config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.

        Returns:

            numpy.ndarray: Phase deviation matrix with axes (frequency_bins, frames).

        Raises:

            AudioError: If matrix computation fails.
        """
    def piptrack(self, /, *, n_fft: int = ..., hop_length: int | None = None, fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", ref: Any | None = None) -> "tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]":
        """
        Spectral peak tracking with parabolic interpolation.

        Args:

            n_fft (int): FFT length. Default 2048.
            hop_length (Optional[int]): None = ``win_length // 4``.
            fmin (float): Lowest reported frequency (Hz). Default 150.
            fmax (float): Peaks must lie below this frequency (Hz). Default 4000.
            threshold (float): Relative threshold. Default 0.1.
            win_length, window, center, pad_mode: STFT parameters.
            ref (None | float | numpy.max | numpy.mean): Threshold reference; None
                (default) = per-frame maximum, a number = absolute threshold.

        Returns:

            tuple: ``(pitches, magnitudes)``, each numpy.ndarray[float32 | float64] of shape
            ``(channels, 1 + n_fft // 2, n_frames)``; zero except at detected peaks.

        Raises:

            ValueError: For invalid parameters.
        """
    def pitch_shift(self, /, n_steps: float, *, bins_per_octave: int = ..., res_type: Any | None = None, scale: bool = False, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> AudioSamples:
        """
        Shifts the pitch by ``n_steps`` without changing the duration.

        Time-stretch by ``rate = 2 ** (-n_steps / bins_per_octave)``, resample by
        ``rate``, then cut/pad to the original length.

        Args:

            n_steps (float): Shift in steps (semitones by default); fractional
                and negative values allowed.
            bins_per_octave (int): Steps per octave. Default 12.
            res_type (str | ResamplingQuality): Quality of this library's
                windowed-sinc resampler: ``"soxr_hq"`` (default) maps to
                ``ResamplingQuality.high``, ``"soxr_mq"`` to ``medium`` and
                ``"soxr_lq"`` to ``fast``.
            scale (bool): Must be False.
            n_fft, hop_length, win_length, window, center, pad_mode: STFT
                parameters of the time stretch.

        Returns:

            AudioSamples: the shifted audio, same length, type and rate.

        Raises:

            ValueError: For a non-finite ``n_steps``, ``scale=True`` or invalid
                parameters.
        """
    def pitch_shift_in_place(self, /, n_steps: float, *, bins_per_octave: int = ..., res_type: Any | None = None, scale: bool = False, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> None:
        """
        In-place ``pitch_shift``; arguments as ``pitch_shift``.
        """
    def plp(self, /, *, hop_length: int = ..., win_length: int = ..., tempo_min: float | None = ..., tempo_max: float | None = ..., prior: Any | None = None) -> "NDArray[Any]":
        """
        Predominant local pulse of every channel.

        Args:

            hop_length (int): Hop of the (median-aggregated) onset envelope.
                Default 512.
            win_length (int): Fourier tempogram window in frames. Default 384.
            tempo_min (Optional[float]): Lowest tempo (BPM). Default 30.
            tempo_max (Optional[float]): Highest tempo (BPM). Default 300.
            prior: Tempo prior (see ``tempo``).

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_frames)``, values in
            ``[0, 1]``; index ``[0]`` for the ``(n_frames,)`` array of mono audio.

        Raises:

            ValueError: If ``tempo_max <= tempo_min`` or parameters are invalid.
        """
    def poly_features(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", order: int = ...) -> "NDArray[Any]":
        """
        Coefficients of a least-squares polynomial fit to every
        magnitude-spectrum frame against the bin frequencies.

        Args:

            n_fft, hop_length (default 512), win_length, window, center, pad_mode:
                As ``spectral_centroid_frames``.
            order (int): Polynomial order. Default 1.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, order + 1, n_frames)``,
            highest power first.

        Raises:

            ValueError: For invalid parameters.
        """
    def power_spectral_density(self, /, window_size: "int" = 2048, overlap: "float" = 0.5) -> Psd:
        """
        Estimate the power spectral density across overlapping windows.

        Args:

            window_size (int): STFT window length in samples; must be greater than zero.
            overlap (float): Fractional overlap between successive windows.

        Returns:

            Psd: Power spectral density carrying both the frequency axis and the
                power-per-Hz density, averaged over time.

        Raises:

            ValueError: If window_size is zero.
            AudioError: If the PSD computation fails.
        """
    def preemphasis(self, /, *, coef: float = 0.97) -> AudioSamples:
        """
        Pre-emphasis ``y[n] = x[n] - coef * x[n - 1]`` of every channel, with
        initial state ``2 x[0] - x[1]``. Use ``audio_samples.effects.preemphasis``
        for ``zi`` / ``return_zf``.

        Args:

            coef (float): Coefficient. Default 0.97.

        Returns:

            AudioSamples: the filtered audio.

        Raises:

            ValueError: For a non-finite ``coef`` or fewer than two samples.
        """
    def preemphasis_in_place(self, /, *, coef: float = 0.97) -> None:
        """
        In-place ``preemphasis``.
        """
    def pseudo_cqt(self, /, *, hop_length: int = ..., fmin: float | None = None, n_bins: int | None = ..., bins_per_octave: int = ..., tuning: float | None = ..., filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
        """
        Pseudo constant-Q magnitude: constant-Q filters applied to a single STFT.

        Args:

            hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode: As ``cqt``.

        Returns:

            numpy.ndarray[float32 | float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def pyin(self, /, *, fmin: float, fmax: float, frame_length: int = ..., hop_length: int | None = None, n_thresholds: int = ..., beta_parameters: tuple[float, float] = ..., boltzmann_parameter: float = 2.0, resolution: float = 0.1, max_transition_rate: float = 35.92, switch_prob: float = 0.01, no_trough_prob: float = 0.01, fill_na: float | None = ..., center: bool = True, pad_mode: str = "constant", transition_min_prob: float | None = ...) -> "tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]":
        """
        Probabilistic YIN with HMM decoding: f0 tracking with voicing decisions.

        Args:

            fmin, fmax (float): Search range in Hz. Required.
            frame_length (int): Default 2048.
            hop_length (Optional[int]): None = ``frame_length // 4``.
            n_thresholds (int): Default 100.
            beta_parameters (tuple[float, float]): Beta prior. Default (2, 18).
            boltzmann_parameter (float): Default 2.
            resolution (float): Pitch-bin width in semitones. Default 0.1.
            max_transition_rate (float): Octaves per second. Default 35.92.
            switch_prob (float): Voiced/unvoiced switch probability. Default 0.01.
            no_trough_prob (float): Default 0.01.
            fill_na (Optional[float]): f0 of unvoiced frames: NaN (default), None
                (the best-guess pitch) or a number.
            center (bool): Default True.
            pad_mode (str): Default ``"constant"``.
            transition_min_prob (Optional[float]): Viterbi pruning threshold
                (None or 0 = exact decoding). Default 1e-4.

        Returns:

            tuple: ``(f0, voiced_flag, voiced_prob)``, each of shape
            ``(channels, n_frames)``: Hz, ``bool``, values in ``[0, 1]``; the float
            arrays are float32 for uint8, int16, int24 and float32 audio and
            float64 otherwise.

        Raises:

            ValueError: For invalid parameters or a too-short uncentred signal.
        """
    def reassigned_spectrogram(self, /, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, reassign_frequencies: bool = True, reassign_times: bool = True, ref_power: float | Callable[..., Any] = ..., fill_nan: bool = False, clip: bool = True, pad_mode: str = "constant") -> "tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]":
        """
        Time-frequency reassigned spectrogram: the instantaneous frequency and
        group-delay time of every bin.

        Args:

            n_fft (int): FFT length. Default 2048.
            hop_length (Optional[int]): None = ``win_length // 4``.
            win_length (Optional[int]): None = ``n_fft``.
            window: Window specification. Default ``"hann"``.
            center (bool): Centre frames. Default True.
            reassign_frequencies (bool): Reassign frequencies. Default True.
            reassign_times (bool): Reassign times. Default True.
            ref_power (float | numpy.max): Power threshold below which estimates are
                NaN. Default 1e-6.
            fill_nan (bool): Replace NaN estimates by bin / frame centres. Default False.
            clip (bool): Clip to ``[0, sr / 2]`` and ``[0, duration]``. Default True.
            pad_mode (str): Padding for ``center``. Default ``"constant"``.

        Returns:

            tuple: ``(freqs, times, mags)``, each numpy.ndarray[float32 | float64] of shape
            ``(channels, 1 + n_fft // 2, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def remix(self, /, intervals: "ArrayLike", *, align_zeros: bool = False) -> AudioSamples:
        """
        Concatenates sample intervals in the given order.

        Args:

            intervals (array_like): ``(n, 2)`` ``[start, end)`` sample ranges, in
                output order; clipped to the signal, empty ranges skipped.
            align_zeros (bool): Snap boundaries to the nearest zero crossing of
                the (mono downmix of the) signal. Default False.

        Returns:

            AudioSamples: the remixed audio.

        Raises:

            ValueError: For malformed or all-empty intervals.
        """
    def remix_in_place(self, /, intervals: "ArrayLike", *, align_zeros: bool = False) -> None:
        """
        In-place ``remix``.
        """
    def remove_dc_offset(self, /) -> None:
        """
        Remove the DC offset from the signal in-place.

        Modifies the signal in place and returns ``None`` (the NumPy array it
        was created from, if any, is left unchanged).

        Raises:

            AudioError: If offset removal fails.
        """
    def repeat(self, /, count: "int") -> AudioSamples:
        """
        Repeat the buffer a fixed number of times.

        Args:

            count (int): Number of repetitions; must be greater than zero.

        Returns:

            AudioSamples: Concatenated buffer containing the repeated audio.

        Raises:

            AudioError: If repetition fails.
        """
    def resample(self, /, target_sample_rate: "int", quality: "ResamplingQuality | str" = ...) -> AudioSamples:
        """
        Resample the audio to a new sample rate.

        A Kaiser-windowed sinc interpolator evaluated at the exact rational
        position of every output sample: the output is time-aligned with the
        input (no filter delay, nothing to trim), the signal is treated as zero
        outside its ends (so the tail is flushed), and the result has exactly
        ``ceil(n * target_sample_rate / sample_rate)`` samples per channel. The
        cut-off sits just below the Nyquist frequency of the lower rate, so
        downsampling is anti-aliased and upsampling anti-imaged. Processing is
        in the sample type's float (float32 for uint8, int16, int24 and float32;
        float64 for int32 and float64); the result keeps the input's sample type.

        Quality tiers (Nyquist of the lower rate):

        * ``"fast"`` (default): downsampling with a short windowed sinc
          (passband 0.80 x Nyquist, 70 dB stopband); upsampling by linear
          interpolation (images are not suppressed).
        * ``"medium"``: passband 0.90 x Nyquist, 100 dB stopband.
        * ``"high"``: passband 0.913 x Nyquist, 125 dB stopband.
        * ``"very_high"``: 185 dB stopband; passband 0.95 x Nyquist for
          decimation by a power of two, 0.98 for other ratios with no prime
          factor above 7, 0.996 for the rest (such as pitch shifts).

        Args:

            target_sample_rate (int): Output sample rate in Hz (> 0).
            quality (ResamplingQuality | str): ``ResamplingQuality.fast`` (default),
                ``.medium``, ``.high``, ``.very_high``, or the names ``"fast"``,
                ``"medium"``, ``"high"``, ``"very_high"``.

        Returns:

            AudioSamples: the resampled audio (a copy when the rates are equal).

        Raises:

            ValueError: If ``target_sample_rate`` is not a positive integer, or an
                unknown quality name is given.
        """
    def resample_by_ratio(self, /, ratio: "float", quality: "ResamplingQuality | str" = ...) -> AudioSamples:
        """
        Resample the audio by a ratio of sample rates.

        The target rate is ``round(sample_rate * ratio)``; the conversion is then
        exactly ``resample(target_rate, quality)`` (delay-free, exactly
        ``ceil(n * target_rate / sample_rate)`` samples; see ``resample`` for the
        quality tiers).

        Args:

            ratio (float): Output rate / input rate (> 0).
            quality (ResamplingQuality | str): As ``resample``; default ``"fast"``.

        Returns:

            AudioSamples: the resampled audio at ``round(sample_rate * ratio)`` Hz.

        Raises:

            ValueError: If ``ratio`` is not positive or the target rate rounds to 0.
        """
    def reverse(self, /) -> AudioSamples:
        """
        Reverse the samples and return a new buffer.

        Returns:

            AudioSamples: Buffer containing the reversed samples.
        """
    def reverse_in_place(self, /) -> None:
        """
        Reverse the samples in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If reversal fails.
        """
    def rfft(self, /, n_fft: "int") -> "NDArray[np.float64]":
        """
        Compute the magnitude of the one-sided FFT.

        Equal to ``numpy.abs(numpy.fft.rfft(x, n_fft))`` per channel: real
        magnitudes of the ``n_fft // 2 + 1`` non-negative frequency bins (use
        :meth:`fft` for the complex values).

        Args:

            n_fft (int): FFT size in samples; must be greater than zero.

        Returns:

            numpy.ndarray: Magnitude spectrum shaped (channels, n_fft // 2 + 1).

        Raises:

            ValueError: If n_fft is zero.
            AudioError: If the RFFT computation fails.
        """
    def rms(self, /) -> float:
        """
        Compute the root mean square (RMS) energy of the samples.

        Returns:

            float: RMS amplitude across all samples and channels.
        """
    def rms_and_peak(self, /) -> tuple[float, Any]:
        """
        Compute the RMS and peak absolute value in a single pass.

        Equivalent to calling ``rms`` and ``peak`` separately, but reads the
        sample buffer only once.

        Returns:

            tuple[float, int | float]: A ``(rms, peak)`` pair where ``rms`` is the
                root-mean-square as a float and ``peak`` is the maximum absolute
                sample value in the native sample type.
        """
    def rms_envelope(self, /, window_size: "int", hop_size: "int") -> "NDArray[Any]":
        """
        Compute a root-mean-square envelope over sliding windows.

        Args:

            window_size (int): Analysis window size in samples; must be greater than zero.
            hop_size (int): Step between windows in samples; must be greater than zero.

        Returns:

            numpy.ndarray: RMS envelope shaped (channels, frames).

        Raises:

            ValueError: If window_size or hop_size is zero.
            AudioError: If envelope computation fails.
        """
    def rms_frames(self, /, *, frame_length: int = ..., hop_length: int = ..., center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
        """
        Frame-wise root-mean-square energy.

        Unlike the scalar ``rms()``, this returns one value per frame.

        Args:

            frame_length (int): Samples per frame. Default 2048.
            hop_length (int): Samples between frames. Default 512.
            center (bool): Centre frames (pad ``frame_length // 2``). Default True.
            pad_mode (str): Padding for ``center``. Default ``"constant"``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``.

        Raises:

            ValueError: For invalid parameters or a too-short uncentred signal.
        """
    @property
    def sample_rate(self, /) -> int:
        """
        Sample rate in Hz.

        Returns:

            int: Sample rate (e.g., 44100, 48000, 96000).

        Examples:

            >>> audio.sample_rate
            44100
        """
    def samples_per_channel(self, /) -> int:
        """
        Number of samples per channel.

        For multi-channel audio, this is the length of each channel.
        For mono audio, this equals the total number of samples.

        Returns:

            int: Number of samples in each channel.

        Examples:

            >>> stereo = aus.AudioSamples.from_array(np.zeros((2, 44100)), 44100)
            >>> stereo.samples_per_channel()
            44100
        """
    def scale(self, /, factor: "float") -> None:
        """
        Scale samples by a constant factor in-place.

        Modifies the signal in place and returns ``None`` (the NumPy array it
        was created from, if any, is left unchanged).

        Args:

            factor (float): Multiplier applied to every sample; values above 1.0 amplify and below 1.0 attenuate.
        """
    @property
    def shape(self, /) -> list[int]:
        """
        Shape of the audio data.

        Returns:

            list[int]: Shape as [samples] for mono or [channels, samples] for multi-channel.

        Examples:

            >>> mono.shape
            [44100]
            >>> stereo.shape
            [2, 44100]
        """
    @property
    def size(self, /) -> int:
        """
        Total number of samples across all channels (NumPy-style alias of ``len``).
        """
    def spectral_bandwidth(self, /, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral bandwidth (spectral spread) of the signal.

        The magnitude-weighted standard deviation of the spectrum about its
        spectral centroid. Larger values indicate energy spread across a wider
        frequency range; a pure tone yields a value near zero.

        Args:

            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Spectral bandwidth in Hz. Returns 0.0 for silence.

        Raises:

            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectral_bandwidth_frames(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", norm: bool = True, p: float = 2.0) -> "NDArray[Any]":
        """
        Order-``p`` spectral bandwidth of every frame around the spectral centroid.

        Args:

            n_fft, hop_length (default 512), win_length, window, center, pad_mode:
                As ``spectral_centroid_frames``.
            norm (bool): Normalise each frame to unit sum. Default True.
            p (float): Order of the deviation. Default 2.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz.

        Raises:

            ValueError: For invalid parameters.
        """
    def spectral_centroid(self, /, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral centroid of the whole signal.

        Args:

            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Centroid in Hz of the **power** spectrum ``|X|**2`` of the whole
                signal (a single FFT, not frame-wise). Returns 0.0 for silence.

        Raises:

            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectral_centroid_frames(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
        """
        Spectral centroid of every frame: the magnitude-weighted mean frequency.

        Computed from the magnitude STFT. For several descriptors of one
        spectrogram use ``features.spectral_centroid_from_spectrogram``.

        Args:

            n_fft (int): FFT length. Default 2048.
            hop_length (int): Samples between frames. Default 512.
            win_length (Optional[int]): Window length; None = ``n_fft``.
            window: Window specification (see ``stft_matrix``). Default ``"hann"``.
            center (bool): Centre frames. Default True.
            pad_mode (str): Padding for ``center``. Default ``"constant"``.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz, also
            for mono audio (index ``[0]`` for ``(1, n_frames)``).

        Raises:

            ValueError: For invalid parameters or a too-short uncentred signal.
        """
    def spectral_contrast(self, /, n_bands: "int", reduction: "ChannelReduction" = ...) -> list[float]:
        """
        Compute spectral contrast across octave-spaced sub-bands.

        The spectrum is partitioned into ``n_bands`` octave-spaced sub-bands.
        Within each band the contrast is the dB difference between the mean of the
        top quantile (peaks) and the mean of the bottom quantile (valleys). High
        contrast indicates clear tonal/harmonic structure.

        Args:

            n_bands (int): Number of octave-spaced sub-bands; must be greater than zero.
            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            list[float]: ``n_bands`` contrast values in dB, low band to high band.

        Raises:

            ValueError: If n_bands is zero.
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectral_contrast_frames(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", fmin: float = 200.0, n_bands: int = ..., quantile: float = 0.02, linear: bool = False) -> "NDArray[Any]":
        """
        Octave-band spectral contrast of every frame: the peak-to-valley
        difference in each band, in dB.

        Args:

            n_fft, hop_length (default 512), win_length, window, center, pad_mode:
                As ``spectral_centroid_frames``.
            fmin (float): Upper edge of the lowest band (Hz). Default 200.
            n_bands (int): Number of octave bands. Default 6.
            quantile (float): Fraction used for peak / valley. Default 0.02.
            linear (bool): Linear difference instead of dB. Default False.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_bands + 1, n_frames)``.

        Raises:

            ValueError: For invalid parameters (e.g. the top band above Nyquist).
        """
    def spectral_crest(self, /, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral crest factor of the signal.

        The ratio of the peak magnitude to the mean magnitude of the spectrum.
        High values indicate a strongly peaked (tonal) spectrum; a flat spectrum
        approaches 1.

        Args:

            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Spectral crest factor (>= 1 for non-silent signals). Returns
                0.0 for silence.

        Raises:

            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectral_flatness(self, /, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral flatness (Wiener entropy) of the signal.

        The ratio of the geometric mean to the arithmetic mean of the power
        spectrum. Values near 1 indicate noise-like (flat) spectra, while values
        near 0 indicate tonal spectra dominated by a few peaks.

        Args:

            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Spectral flatness in [0, 1]. Returns 0.0 for silence.

        Raises:

            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectral_flatness_frames(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", amin: float = 1e-10, power: float = 2.0) -> "NDArray[Any]":
        """
        Spectral flatness of every frame: the geometric over the arithmetic mean
        of the power spectrum.

        Args:

            n_fft, hop_length (default 512), win_length, window, center, pad_mode:
                As ``spectral_centroid_frames``.
            amin (float): Floor of ``|S| ** power``. Default 1e-10.
            power (float): Exponent of the magnitude. Default 2.0.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``, in ``[0, 1]``.

        Raises:

            ValueError: For invalid parameters.
        """
    def spectral_flux(self, /, cqt_params: "CqtParams", window_size: "int", hop_size: "int", method: "SpectralFluxMethod") -> tuple[list[float], list[float]]:
        """
        Compute spectral flux values without peak picking.

        Args:

            cqt_params (CqtParams): Parameters controlling the constant-Q transform.
            window_size (int): Frame size in samples; must be greater than zero.
            hop_size (int): Hop size in samples; must be greater than zero.
            method (SpectralFluxMethod): Flux computation strategy.

        Returns:

            tuple[list[float], list[float]]: ``(times, flux)``: the frame times in
                seconds first, then the flux values (one per frame).

        Raises:

            ValueError: If window_size or hop_size is zero.
            AudioError: If flux computation fails.
        """
    def spectral_rolloff(self, /, rolloff_percent: "float" = 0.85, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral rolloff point for the provided energy fraction.

        Args:

            rolloff_percent (float): Target cumulative spectral energy fraction between 0 and 1.
            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Frequency where the cumulative spectrum reaches the rolloff threshold.

        Raises:

            AudioError: If the FFT computation fails or rolloff_percent is invalid.
        """
    def spectral_rolloff_frames(self, /, *, n_fft: int = ..., hop_length: int = ..., win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", roll_percent: float = 0.85) -> "NDArray[Any]":
        """
        Roll-off frequency of every frame: the frequency below which
        ``roll_percent`` of the energy lies.

        Args:

            n_fft, hop_length (default 512), win_length, window, center, pad_mode:
                As ``spectral_centroid_frames``.
            roll_percent (float): Energy fraction in ``(0, 1)``. Default 0.85.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz.

        Raises:

            ValueError: For invalid parameters.
        """
    def spectral_slope(self, /, reduction: "ChannelReduction" = ...) -> float:
        """
        Compute the spectral slope of the signal.

        The slope of an ordinary least-squares linear fit of the linear magnitude
        spectrum against frequency (Hz), in units of magnitude per Hz. A negative
        slope indicates energy concentrated at low frequencies.

        Args:

            reduction (ChannelReduction): Channel-reduction policy applied to
                multi-channel input. Defaults to ``ChannelReduction.average``.

        Returns:

            float: Least-squares slope (magnitude per Hz). Returns 0.0 for
                silence or a degenerate single-bin spectrum.

        Raises:

            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
    def spectrogram_matrix(self, /, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", power: float = 1.0) -> "NDArray[Any]":
        """
        Magnitude spectrogram ``|STFT| ** power`` of every channel.

        Equivalent to ``numpy.abs(self.stft_matrix(...)) ** power``.

        Args:

            n_fft, hop_length, win_length, window, center, pad_mode: As ``stft_matrix``.
            power (float): Exponent; 1.0 (default) gives magnitude, 2.0 power.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1 + n_fft // 2, n_frames)``;
            float32 for uint8, int16, int24 and float32 audio, float64 for int32
            and float64 audio.

        Raises:

            ValueError: For invalid parameters or a non-positive power.
        """
    def speech_regions(self, /, config: "VadConfig") -> list[tuple[int, int]]:
        """
        Compute contiguous speech regions as (``start_sample``, ``end_sample``) pairs.

        Args:

            config (VadConfig): Parameters controlling the voice activity detector.

        Returns:

            list[tuple[int, int]]: Regions marked as speech with sample-based start and exclusive end.

        Raises:

            AudioError: If region extraction fails.
        """
    def split(self, /, segment_duration_seconds: "float") -> list[AudioSamples]:
        """
        Split the buffer into equal-length segments.

        Args:

            segment_duration_seconds (float): Duration of each segment.

        Returns:

            list[AudioSamples]: List of segments that cover the original buffer.

        Raises:

            AudioError: If segmentation fails or the duration is invalid.
        """
    def split_db(self, /, *, top_db: float, ref: float | Literal["max"] | Callable[..., Any] = ..., frame_length: int = ..., hop_length: int = ..., aggregate: Any | None = None) -> "NDArray[np.int64]":
        """
        Non-silent intervals: frames within ``top_db`` of the reference.

        Args:

            top_db, ref, frame_length, hop_length, aggregate: As ``trim_db``.

        Returns:

            numpy.ndarray[int64]: shape ``(n_intervals, 2)`` of ``[start, end)``
            sample ranges (``(0, 2)`` if everything is silent).
        """
    @classmethod
    def stack(cls, /, sources: "list[AudioSamples]") -> AudioSamples:
        """
        Stack multiple buffers along the channel axis.

        Args:

            sources (list[AudioSamples]): Buffers to stack; must share dtype and sample rate.

        Returns:

            AudioSamples: Combined buffer with channels from each source.

        Raises:

            AudioError: If stacking fails or sources are incompatible.
        """
    def std_dev(self, /) -> float:
        """
        Compute the standard deviation of the sample values.

        Returns:

            float: Standard deviation derived from the variance.
        """
    def stft(self, /, params: "StftParams") -> "StftResult":
        """
        Compute the Short-Time Fourier Transform (STFT).

        Args:

            params (StftParams): Configuration controlling window length, hop size, and window type.

        Returns:

            StftResult: Object containing complex spectrum, frequency bins, and time stamps.

        Raises:

            AudioError: If the STFT computation fails.
        """
    def stft_matrix(self, /, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
        """
        Complex short-time Fourier transform of every channel.

        Args:

            n_fft (int): FFT length in samples. Default 2048.
            hop_length (Optional[int]): Samples between frames. ``None`` means
                ``win_length // 4``.
            win_length (Optional[int]): Window length (<= n_fft); ``None`` means
                ``n_fft``. Shorter windows are zero-padded to ``n_fft``.
            window (str | tuple | numpy.ndarray): Window specification, as
                ``scipy.signal.get_window``: ``"hann"`` (default), ``"hamming"``,
                ``"blackman"``, ``"blackmanharris"``, ``"boxcar"``/``"rectangular"``,
                ``"bartlett"``, ``"triang"``, ``("kaiser", beta)``,
                ``("gaussian", std)``, or an array of ``win_length`` coefficients.
                Named windows are periodic (``fftbins=True``).
            center (bool): Pad by ``n_fft // 2`` on both sides so frame ``t`` is
                centred on sample ``t * hop_length``. Default True.
            pad_mode (str): ``"constant"`` (zeros, default), ``"reflect"``,
                ``"edge"`` or ``"symmetric"``.

        Returns:

            numpy.ndarray[complex64 | complex128]: shape ``(channels, 1 + n_fft // 2, n_frames)``;
            complex64 for uint8, int16, int24 and float32 audio, complex128 for
            int32 and float64 audio.
            Mono audio gives a leading axis of length 1 (index ``[0]`` for
            ``(1 + n_fft // 2, n_frames)``).

        Raises:

            ValueError: For invalid parameters, or ``center=False`` with a signal
                shorter than ``n_fft``.
        """
    def sub_numpy_array(self, /, numpy_array: "NDArray[Any]") -> AudioSamples: ...
    def swap_channels(self, /, channel1: "int", channel2: "int") -> None:
        """
        Swap two channels in place.

        Modifies the signal in place and returns ``None``. If the ``AudioSamples``
        was created from a NumPy array (``new_mono``/``new_multi``), that array
        is modified too; pass ``arr.copy()`` to keep the original.

        Args:

            channel1 (int): Zero-based index of the first channel.
            channel2 (int): Zero-based index of the second channel.

        Returns:

            None: Operation mutates the current buffer.

        Raises:

            AudioError: If swapping fails or indices are invalid.
        """
    def tempo(self, /, *, hop_length: int = ..., start_bpm: float = 120.0, std_bpm: float = 1.0, ac_size: float = 8.0, max_tempo: float | None = ..., aggregate: "Callable[..., Any] | Literal['mean', 'median', 'max', 'min', 'sum']" = ..., prior: Any | None = None) -> "NDArray[np.float64]":
        """
        Estimates the tempo of every channel in BPM.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            start_bpm (float): Prior centre. Default 120.
            std_bpm (float): Prior spread in octaves. Default 1.
            ac_size (float): Autocorrelation window in seconds. Default 8.
            max_tempo (Optional[float]): Exclude tempi at or above this. Default 320.
            aggregate: Reduction over time (default ``numpy.mean``); None gives
                one tempo per frame.
            prior: None (log-normal), a ``scipy.stats`` distribution or a callable
                of the BPM returning a log-density.

        Returns:

            numpy.ndarray[float64]: shape ``(channels, 1)`` (``(channels, n_frames)``
            with ``aggregate=None``); index ``[0]`` for the ``(1,)`` array of
            mono audio.

        Raises:

            ValueError: For invalid parameters.
        """
    def tempogram(self, /, *, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ..., norm: float | None = ...) -> "NDArray[Any]":
        """
        Autocorrelation tempogram of every channel.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            win_length (int): Autocorrelation window (lag bins). Default 384.
            center (bool): Centre the windows. Default True.
            window: Window before the autocorrelation. Default ``"hann"``.
            norm (Optional[float]): Column normalisation (``numpy.inf`` default,
                a p-norm, or None).

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, win_length, n_frames)``;
            index ``[0]`` for the 2-D array of mono audio.

        Raises:

            ValueError: For invalid parameters.
        """
    def tempogram_ratio(self, /, *, hop_length: int = ..., win_length: int = ..., start_bpm: float = 120.0, std_bpm: float = 1.0, max_tempo: float | None = ..., factors: Any | None = None, aggregate: Any | None = None, prior: Any | None = None, center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ..., fill_value: float = 0.0, norm: float | None = ...) -> "NDArray[Any]":
        """
        Tempogram ratio features of every channel: the tempogram sampled at
        fixed multiples of the per-frame tempo estimate.

        Args:

            hop_length (int): Hop of the onset envelope. Default 512.
            win_length, center, window, norm: Tempogram parameters (as ``tempogram``).
            start_bpm, std_bpm, max_tempo, prior: Per-frame tempo estimation (as
                ``tempo``).
            factors (Optional[array_like]): Tempo multiples; None uses the 13
                default factors.
            aggregate: Optional reduction over time (default None).
            fill_value (float): Value outside the tempogram range. Default 0.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, n_factors, n_frames)``, or
            ``(channels, n_factors, 1)`` when aggregated (the time axis is kept);
            index ``[0]`` for the array of mono audio.
        """
    def time_axis(self, /) -> "NDArray[np.float64]":
        """
        Time of each sample in seconds, one value per sample of a channel:
        ``numpy.arange(samples_per_channel()) / sample_rate``.
        """
    def time_stretch(self, /, rate: float, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> AudioSamples:
        """
        Changes the speed without changing the pitch.

        STFT, ``phase_vocoder`` by ``rate``, inverse STFT to exactly
        ``round(n / rate)`` samples.

        Args:

            rate (float): Speed-up factor (2.0 = twice as fast, half the length).
            n_fft, hop_length, win_length, window, center, pad_mode: STFT
                parameters, as ``stft_matrix``.

        Returns:

            AudioSamples: the stretched audio (input sample type and rate).

        Raises:

            ValueError: If ``rate`` is not positive, the result would be empty,
                or STFT parameters are invalid.
        """
    def time_stretch_in_place(self, /, rate: float, *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> None:
        """
        In-place ``time_stretch`` (the length changes); arguments as ``time_stretch``.
        """
    def to_format(self, /, dtype: SampleType) -> AudioSamples:
        """
        Convert to another sample type with audio-aware scaling.

        Audio-aware conversion: values are rescaled between the formats'
        full-scale ranges, so a float ``1.0`` and an int16 ``32767`` both mean
        full scale. Integer scaling is asymmetric because two's-complement
        ranges are: float ``-1.0`` maps to ``-32768`` and ``+1.0`` to ``32767``
        (int16; likewise for int24/int32), i.e. negative values scale by
        ``2**(bits-1)`` and positive values by ``2**(bits-1) - 1``, with
        rounding and saturation at the integer limits. uint8 is offset binary
        (``-1.0 -> 0``, ``0.0 -> 128``, ``1.0 -> 255``). Float-to-float
        conversions do not clip. Compare :meth:`cast_as`, which keeps the
        numeric value instead.

        Args:

            dtype (SampleType): Target type, e.g. ``SampleType.I16``.

        Returns:

            AudioSamples: New buffer of the target type.

        Examples:

            >>> import numpy as np, audio_samples as aus
            >>> x = aus.AudioSamples.new_mono(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), sample_rate=8000)
            >>> x.to_format(aus.SampleType.I16).to_numpy()
            array([-32768, -16384,      0,  16384,  32767], dtype=int16)
        """
    def to_gpu_tensor(self, /, gpu_id: "Optional[str]" = None) -> Any:
        """
        Convert ``AudioSamples`` to a PyTorch tensor on the GPU if possible
        """
    def to_mono(self, /, method: "MonoConversionMethod" = ...) -> AudioSamples:
        """
        Convert the audio buffer to mono using the provided strategy.

        Args:

            method (MonoConversionMethod): Conversion method describing how to combine channels.

        Returns:

            AudioSamples: New buffer with a single channel.

        Raises:

            AudioError: If channel conversion fails.
        """
    def to_numpy(self, /) -> "NDArray[Any]":
        """
        Convert audio samples to a NumPy array.

        Returns the audio data as a numpy.ndarray. If the audio is already
        backed by a numpy array, returns the existing array (zero-copy).
        Otherwise, creates a new numpy array from the owned data.

        Returns:

            numpy.ndarray: Audio data with shape:
                - (samples,) for mono audio
                - (channels, samples) for multi-channel audio

        Examples:

            >>> audio = aus.AudioSamples.from_array(np.random.randn(2, 44100), 44100)
            >>> arr = audio.to_numpy()
            >>> arr.shape
            (2, 44100)
            >>> arr.dtype
            dtype('float64')
        """
    def to_stereo(self, /, method: "StereoConversionMethod" = ...) -> AudioSamples:
        """
        Convert the audio buffer to stereo using the provided strategy.

        Args:

            method (StereoConversionMethod): Conversion method describing how to expand channels.

        Returns:

            AudioSamples: New buffer with two channels.

        Raises:

            AudioError: If channel conversion fails.
        """
    def to_tensor(self, /, device: "Optional[str]") -> Any:
        """
        Convert ``AudioSamples`` to a PyTorch tensor
        """
    def tonnetz(self, /, *, hop_length: int = ..., fmin: float | None = None, norm: float | None = ..., threshold: float | None = ..., tuning: float | None = ..., n_chroma: int = ..., n_octaves: int = ..., bins_per_octave: int | None = ..., cqt_mode: str = "full") -> "NDArray[Any]":
        """
        Tonal centroid features (tonnetz) from the constant-Q chroma.

        Args:

            hop_length, fmin, norm, threshold, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode:
                ``chroma_cqt`` parameters.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 6, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    @property
    def total_samples(self, /) -> int:
        """
        Total number of samples across all channels.

        Returns:

            int: Total samples = num_channels x samples_per_channel.

        Examples:

            >>> stereo.total_samples  # 2 channels x 44100 samples
            88200
        """
    def track_pitch(self, /, window_size: "int", hop_size: "int", pitch_method: "PitchDetectionMethod" = ..., threshold: "float" = 0.1, min_frequency: "float" = 50.0, max_frequency: "float" = 2000.0) -> PitchContour:
        """
        Track the detected pitch over successive analysis windows.

        Args:

            window_size (int): Size of each analysis window in samples; must be greater than zero.
            hop_size (int): Step between windows in samples; must be greater than zero.
            pitch_method (PitchDetectionMethod): Detection strategy to apply per
                window. Only ``yin`` and ``autocorrelation`` are implemented;
                ``cepstrum`` and ``harmonic_product`` raise ``AudioParameterError``.
            threshold (float): YIN threshold on d'(tau) in [0, 1]; lower is
                stricter (more unvoiced frames). Ignored for autocorrelation.
            min_frequency (float): Minimum expected frequency in hertz.
            max_frequency (float): Maximum expected frequency in hertz.

        Returns:

            PitchContour: Time-ordered pitch track of per-window frames.

        Raises:

            ValueError: If window_size or hop_size is zero.
            AudioError: If pitch tracking fails.
        """
    def trim(self, /, start_seconds: "float", end_seconds: "float") -> AudioSamples:
        """
        Trim the buffer to the given time range.

        Args:

            start_seconds (float): Inclusive start offset in seconds.
            end_seconds (float): Exclusive end offset in seconds.

        Returns:

            AudioSamples: Buffer containing only the requested region.

        Raises:

            AudioError: If the range is invalid or trimming fails.
        """
    def trim_all_silence(self, /, threshold_db: "float", min_silence_duration_seconds: "float") -> AudioSamples:
        """
        Remove every silent region longer than a minimum duration.

        Unlike :meth:`trim_silence`, which only trims leading and trailing
        silence, this removes interior silence runs as well. A region is treated
        as silent when its level stays below ``threshold_db`` for at least
        ``min_silence_duration_seconds``.

        Args:

            threshold_db (float): Silence threshold in decibels.
            min_silence_duration_seconds (float): Minimum duration, in seconds, a
                silent run must reach before it is removed.

        Returns:

            AudioSamples: Buffer with qualifying silence removed.

        Raises:

            AudioError: If trimming fails.
        """
    def trim_db(self, /, *, top_db: float, ref: float | Literal["max"] | Callable[..., Any] = ..., frame_length: int = ..., hop_length: int = ..., aggregate: Any | None = None) -> tuple[AudioSamples, Incomplete]:
        """
        Trims leading and trailing silence.

        A frame is non-silent when its RMS level (centred frames, dB relative to
        ``ref``) is above ``-top_db``; the kept region runs from the first
        non-silent frame to the end of the last. Not to be confused with
        ``trim_silence`` (sample-wise absolute dBFS threshold).

        Args:

            top_db (float): Threshold below the reference in dB. Required.
            ref (float | "max" | numpy.max): Reference RMS amplitude; the default
                ``numpy.max`` uses the loudest frame.
            frame_length (int): RMS frame length. Default 2048.
            hop_length (int): Hop between frames. Default 512.
            aggregate: Channel combination: ``numpy.max`` (default), ``numpy.min``,
                ``numpy.mean`` or any callable, called as ``aggregate(db, 0)`` on the
                ``(channels, frames)`` levels in dB of multichannel audio. A callable
                here or in ``ref`` runs with the GIL held.

        Returns:

            tuple: ``(trimmed, index)``: the trimmed ``AudioSamples`` and an int64
            array ``[start, end]`` of the kept sample range.

        Raises:

            ValueError: For invalid parameters, or if everything is silent (an
                ``AudioSamples`` cannot be empty; use ``split_db`` to test).
        """
    def trim_db_in_place(self, /, *, top_db: float, ref: float | Literal["max"] | Callable[..., Any] = ..., frame_length: int = ..., hop_length: int = ..., aggregate: Any | None = None) -> "NDArray[np.int64]":
        """
        In-place ``trim_db``; returns the kept ``[start, end]`` (int64 array).
        """
    def trim_silence(self, /, threshold_db: "float") -> AudioSamples:
        """
        Trim leading and trailing silence below a threshold.

        Args:

            threshold_db (float): Silence threshold in decibels.

        Returns:

            AudioSamples: Buffer with silence removed.

        Raises:

            AudioError: If trimming fails.
        """
    @staticmethod
    def uniform_mono(length: "int", sample_rate: "int", value: "float") -> AudioSamples: ...
    @staticmethod
    def uniform_mono_f64(length: "int", sample_rate: "int", value: "float") -> AudioSamples: ...
    @staticmethod
    def uniform_mono_i16(length: "int", sample_rate: "int", value: "int") -> AudioSamples: ...
    @staticmethod
    def uniform_mono_i32(length: "int", sample_rate: "int", value: "int") -> AudioSamples: ...
    @staticmethod
    def uniform_multi(channels: "int", length: "int", sample_rate: "int", value: "float") -> AudioSamples: ...
    @staticmethod
    def uniform_multi_f64(channels: "int", length: "int", sample_rate: "int", value: "float") -> AudioSamples: ...
    @staticmethod
    def uniform_multi_i16(channels: "int", length: "int", sample_rate: "int", value: "int") -> AudioSamples: ...
    @staticmethod
    def uniform_multi_i32(channels: "int", length: "int", sample_rate: "int", value: "int") -> AudioSamples: ...
    def variance(self, /) -> float:
        """
        Compute the variance of the sample values.

        Returns:

            float: Variance of amplitudes across all samples and channels.
        """
    def voice_activity_mask(self, /, config: "VadConfig") -> list[bool]:
        """
        Compute a per-frame voice activity mask.

        Args:

            config (VadConfig): Parameters controlling the voice activity detector.

        Returns:

            list[bool]: Boolean mask where True marks frames with speech activity.

        Raises:

            AudioError: If mask computation fails.
        """
    def vqt(self, /, *, hop_length: int = ..., fmin: float | None = None, n_bins: int | None = ..., intervals: Any | None = None, gamma: float | None = None, bins_per_octave: int = ..., tuning: float | None = ..., filter_scale: float = 1.0, norm: float | None = ..., sparsity: float = 0.01, window: str | tuple[str, float] | float | ArrayLike = ..., scale: bool = True, pad_mode: str = "constant", res_type: str = "soxr_hq") -> "NDArray[Any]":
        """
        Variable-Q transform of every channel.

        Args:

            hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: As ``cqt``.
            intervals (str): Only ``"equal"`` (equal temperament) is supported.
            gamma (Optional[float]): Bandwidth offset (Hz); None (default) =
                ERB-derived, 0 = constant-Q.

        Returns:

            numpy.ndarray[complex64 | complex128]: shape ``(channels, n_bins, n_frames)``.

        Raises:

            ValueError: For invalid parameters or ``intervals`` other than ``"equal"``.
        """
    def windows(self, /, window_size: "int", hop_size: "int") -> list:
        """
        Split the signal into fixed-size, optionally overlapping windows.

        Each window covers ``window_size`` samples per channel and successive
        windows start ``hop_size`` samples apart, so windows overlap when
        ``hop_size < window_size``. Trailing data that does not fill a complete
        window is zero-padded (the crate's default ``PaddingMode.Zero``).

        Note:

            The crate's zero-copy ``windows_ref`` iterator yields views that
            borrow into the buffer and cannot be exposed to Python safely. This
            method therefore copies each window into a new NumPy array.

        Args:

            window_size (int): Samples per channel in each window; must be > 0.
            hop_size (int): Samples between successive window starts; must be > 0.

        Returns:

            list[numpy.ndarray]: One array per window, each shaped ``(samples,)``
                for mono audio or ``(channels, samples)`` for multi-channel audio.

        Raises:

            ValueError: If window_size or hop_size is zero.
        """
    def yin(self, /, *, fmin: float, fmax: float, frame_length: int = ..., hop_length: int | None = None, trough_threshold: float = 0.1, center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
        """
        Frame-wise YIN fundamental frequency.

        Unlike ``detect_pitch_yin()`` / ``track_pitch()``, this uses centred
        frames, the cumulative mean normalised difference (CMND), parabolic
        refinement of the chosen trough and a fallback to the global minimum
        when no trough is below ``trough_threshold``.

        Args:

            fmin (float): Lowest frequency searched (Hz). Required (e.g. C2 = 65.41).
            fmax (float): Highest frequency searched (Hz). Required (e.g. C7 = 2093.0).
            frame_length (int): Samples per frame. Default 2048.
            hop_length (Optional[int]): None = ``frame_length // 4``.
            trough_threshold (float): Absolute CMND threshold. Default 0.1.
            center (bool): Centre frames on ``t * hop_length``. Default True.
            pad_mode (str): Padding for ``center``. Default ``"constant"``.

        Returns:

            numpy.ndarray[float32 | float64]: f0 in Hz, shape ``(channels, n_frames)``.

        Raises:

            ValueError: For invalid parameters (e.g. ``fmax > sr / 2``) or a
                too-short uncentred signal.
        """
    def zero_crossing_rate(self, /) -> float:
        """
        Calculate the zero-crossing rate of the signal, in crossings per second.

        ``zero_crossing_rate = zero_crossings() / (channels * duration_seconds)``,
        where ``duration_seconds = samples_per_channel / sample_rate``: for
        multi-channel audio it is the mean of the per-channel rates. A pure sine
        of frequency ``f`` gives about ``2 * f``.

        This is one global value in crossings per second; divide by
        ``sample_rate`` for crossings per sample. For the per-frame fraction of
        sign changes use ``zero_crossing_rate_frames``.

        Returns:

            float: Zero crossings **per second** (crossings / duration),
                averaged over channels.
        """
    def zero_crossing_rate_frames(self, /, *, frame_length: int = ..., hop_length: int = ..., center: bool = True, threshold: float = 1e-10, ref_magnitude: Any | None = None, pad: bool = False, zero_pos: bool = True) -> "NDArray[Any]":
        """
        Frame-wise zero-crossing rate (fraction of sign changes per frame).

        Unlike the scalar ``zero_crossing_rate()`` (crossings per second), this
        returns one fraction per frame. Centring always uses edge padding.

        Args:

            frame_length (int): Samples per frame. Default 2048.
            hop_length (int): Samples between frames. Default 512.
            center (bool): Centre frames. Default True.
            threshold (float): ``|x| <= threshold * ref`` counts as zero. Default 1e-10.
            ref_magnitude (None | float | numpy.max): Threshold scale. Default None.
            pad (bool): Count the first sample of every frame as a crossing. Default False.
            zero_pos (bool): Zero counts as positive. Default True.

        Returns:

            numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``.

        Raises:

            ValueError: For invalid parameters.
        """
    def zero_crossings(self, /) -> int:
        """
        Count the number of zero crossings in the signal.

        A crossing is counted between consecutive samples ``x[n-1]`` and ``x[n]``
        when one is ``> 0`` and the other is ``<= 0`` (zero counts as
        non-positive). For multi-channel audio the per-channel counts are summed.

        Returns:

            int: Number of crossings (summed over channels).
        """
    @staticmethod
    def zeros_mono(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_mono_f64(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_mono_i16(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_mono_i32(length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_multi(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_multi_f64(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_multi_i16(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...
    @staticmethod
    def zeros_multi_i32(channels: "int", length: "int", sample_rate: "int") -> AudioSamples: ...

def __getattr__(name: str) -> Incomplete: ...
