"""Array forms and extra options of common operations.

These functions take and return NumPy arrays, and cover options that the methods of
``AudioSamples`` leave out because an ``AudioSamples`` makes them unnecessary: several
signals of different lengths, a reading window, resampling with an energy scale, dense
event masks, and framing along any axis. They build on the operations of this package.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

import audio_samples as _aus

__all__ = [
    "beat_track",
    "chroma_cens",
    "chroma_cqt",
    "chroma_vqt",
    "frame",
    "get_duration",
    "istft",
    "onset_detect",
    "onset_strength",
    "pitch_shift",
    "read_segment",
    "reassigned_spectrogram",
    "resample",
    "salience",
    "stft",
    "tempogram_ratio",
    "stream_blocks",
    "to_mono",
    "to_multi",
    "to_stereo",
]


def _checked(signals: Any) -> list[NDArray[Any]]:
    """Float arrays of at least one dimension with finite samples."""
    arrays = [np.atleast_1d(s) for s in signals]
    for a in arrays:
        _aus.arrays.valid_audio(a)
    return arrays


def _target(out: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    if out.shape != shape:
        raise ValueError(f"out: has shape {out.shape}, expected {shape}")
    out[...] = 0
    return out


def _add(dst: NDArray[Any], a: NDArray[Any], scale: float = 1.0) -> None:
    """Adds ``a`` to the start of ``dst``, each trimmed to the shorter of the two."""
    n = min(dst.shape[-1], a.shape[-1])
    if scale == 1.0:
        dst[..., :n] += a[..., :n]
    else:
        np.add(dst[..., :n], a[..., :n] * scale, out=dst[..., :n], casting="unsafe")


def _mix_into(dst: NDArray[Any], arrays: list[NDArray[Any]], norm: bool) -> None:
    """Sums the channels of each signal into ``dst``; ``norm`` averages channels and signals."""
    if len(arrays) == 1:
        a = arrays[0]
        rows = a.reshape(-1, a.shape[-1])
        for row in rows:
            _add(dst, row)
        if norm and rows.shape[0] > 1:
            dst /= rows.shape[0]
        return
    for a in arrays:
        rows = a.reshape(-1, a.shape[-1])
        scale = 1.0 / rows.shape[0] if norm else 1.0
        for row in rows:
            _add(dst, row, scale)
    if norm:
        dst /= len(arrays)


def to_mono(*signals: NDArray[Any], pad: bool = True, norm: bool = True, out: NDArray[Any] | None = None) -> NDArray[Any]:
    """Mixes the channels of one or more signals to one channel.

    Args:
        *signals: Arrays with time on the last axis.
        pad: Pad shorter signals to the longest; otherwise trim to the shortest.
        norm: Average the channels; otherwise sum them.
        out: Array to write the result to.

    Returns:
        The mixed signal.

    Raises:
        ValueError: No signals, or a wrong ``out`` shape.
    """
    if not signals:
        raise ValueError("signals: at least one is needed")
    arrays = _checked(signals)
    lengths = [a.shape[-1] for a in arrays]
    size = max(lengths) if pad else min(lengths)
    dtype = arrays[0].dtype
    for a in arrays[1:]:
        dtype = np.promote_types(dtype, a.dtype)
    mixed = np.zeros(size, dtype=dtype) if out is None else _target(out, (size,))
    _mix_into(mixed, arrays, norm)
    return mixed


def to_stereo(
    *,
    left: NDArray[Any] | None,
    right: NDArray[Any] | None,
    downmix: bool = True,
    pad: bool = True,
    norm: bool = True,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Combines signals into two channels.

    Args:
        left: The left signal; None leaves it silent.
        right: The right signal; None leaves it silent.
        downmix: Mix each input to one channel first. Otherwise a stereo input is summed with
            the other side.
        pad: Pad to the longer signal; otherwise trim to the shorter.
        norm: With ``downmix``, average each input's channels; otherwise halve the sum of two
            inputs.
        out: Array to write the ``(2, n)`` result to.

    Returns:
        A ``(2, n)`` array.

    Raises:
        ValueError: No inputs, or an input of a shape this cannot combine.
    """
    if left is None and right is None:
        raise ValueError("left: at least one of left and right is needed")
    one_sided = left is None or right is None
    lhs = np.zeros_like(right) if left is None else np.asarray(left)
    rhs = np.zeros_like(left) if right is None else np.asarray(right)
    size = max(lhs.shape[-1], rhs.shape[-1]) if pad else min(lhs.shape[-1], rhs.shape[-1])
    dtype = np.promote_types(lhs.dtype, rhs.dtype)
    output = np.zeros((2, size), dtype=dtype) if out is None else _target(out, (2, size))
    if downmix:
        _mix_into(output[0], [np.atleast_1d(lhs)], norm)
        _mix_into(output[1], [np.atleast_1d(rhs)], norm)
    else:
        for side, a in enumerate((lhs, rhs)):
            if a.ndim == 1:
                _add(output[side], a)
            elif a.ndim == 2 and a.shape[0] == 2:
                _add(output, a)
            else:
                raise ValueError(f"left, right: unsupported shape {a.shape} for downmix=False")
        if norm and not one_sided:
            output /= 2
    return output


def to_multi(
    *signals: NDArray[Any],
    downmix: bool = True,
    pad: bool = True,
    norm: bool = True,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Combines signals into one array with a leading channel axis.

    Args:
        *signals: Arrays with time on the last axis.
        downmix: Give each signal one channel of the result; otherwise mix signals of one
            channel layout, channel by channel.
        pad: Pad to the longest signal; otherwise trim to the shortest.
        norm: With ``downmix``, average each signal's channels; otherwise divide the mix by
            the number of signals.
        out: Array to write the result to.

    Returns:
        The combined array.

    Raises:
        ValueError: No signals, a non-finite signal, or different layouts with ``downmix=False``.
    """
    if not signals:
        raise ValueError("signals: at least one is needed")
    arrays = _checked(signals)
    lengths = [a.shape[-1] for a in arrays]
    size = max(lengths) if pad else min(lengths)
    dtype = arrays[0].dtype
    for a in arrays[1:]:
        dtype = np.promote_types(dtype, a.dtype)
    if downmix:
        shape: tuple[int, ...] = (len(arrays), size)
        output = np.zeros(shape, dtype=dtype) if out is None else _target(out, shape)
        for row, a in zip(output, arrays):
            _mix_into(row, [a], norm)
        return output
    layout = arrays[0].shape[:-1]
    if any(a.shape[:-1] != layout for a in arrays):
        raise ValueError("signals: different channel layouts with downmix=False")
    shape = (*layout, size)
    output = np.zeros(shape, dtype=dtype) if out is None else _target(out, shape)
    for a in arrays:
        _add(output, a)
    if norm:
        output /= len(arrays)
    return output


def get_duration(
    *,
    y: NDArray[Any] | None = None,
    sr: float = 22050,
    S: NDArray[Any] | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    path: str | os.PathLike[str] | None = None,
) -> float:
    """The duration in seconds of samples, of a spectrogram, or of a file.

    Args:
        y: Samples with time on the last axis.
        sr: Sample rate in Hz of ``y`` or ``S``.
        S: A spectrogram with frames on the last axis.
        n_fft: FFT size of ``S``.
        hop_length: Samples between the frames of ``S``.
        center: The frames of ``S`` are centred.
        path: A file, whose header gives the duration.

    Returns:
        The duration.

    Raises:
        ValueError: None of ``y``, ``S`` and ``path`` is given.
    """
    if path is not None:
        return float(_aus.io.info(os.fspath(path)).duration)
    if y is not None:
        return float(np.shape(y)[-1]) / sr
    if S is None:
        raise ValueError("y: give one of y, S or path")
    samples = n_fft + hop_length * (S.shape[-1] - 1)
    if center:
        samples -= 2 * (n_fft // 2)
    return float(samples) / sr


def _native_type(path: str) -> Any:
    name = str(_aus.io.peek_native_type(path)).upper()
    return getattr(_aus.SampleType, name)


def read_segment(path: str | os.PathLike[str], *, offset: float = 0.0, duration: float | None = None, **kwargs: Any) -> Any:
    """Reads part of an audio file.

    Args:
        path: The file.
        offset: Start in seconds.
        duration: Length in seconds. Default: to the end.
        **kwargs: ``as_type`` is the sample type to read as; the file's by default.

    Returns:
        ``AudioSamples`` of the part, with the file's rate and channels. Only that part is
        read.

    Raises:
        ValueError: A negative ``offset`` or ``duration``.
    """
    if offset < 0 or (duration is not None and duration < 0):
        raise ValueError("offset, duration: must not be negative")
    as_type = kwargs.pop("as_type", None)
    if kwargs:
        raise TypeError(f"unexpected arguments {sorted(kwargs)}")
    file = os.fspath(path)
    dtype = as_type if as_type is not None else _native_type(file)
    reader = _aus.io.StreamedAudioReader(file)
    try:
        rate = reader.sample_rate
        start = int(offset * rate)
        available = max(reader.total_frames - start, 0)
        count = available if duration is None else min(int(duration * rate), available)
        if count > 0:
            reader.seek_to_frame(start)
            part = reader.read_frames(count, dtype)
            if part is not None:
                return part
    finally:
        reader.close()
    empty = np.zeros(0, dtype=np.float32)
    return _aus.AudioSamples.new_mono(empty, rate)


def stream_blocks(
    path: str | os.PathLike[str],
    *,
    block_length: int,
    frame_length: int,
    hop_length: int,
    sr: int | None = None,
    mono: bool = False,
    offset: float = 0.0,
    duration: float | None = None,
    fill_value: float | None = None,
    res_type: str = "soxr_hq",
) -> Iterator[NDArray[Any]]:
    """Iterates over blocks of a file, each covering ``block_length`` frames.

    Args:
        path: The file.
        block_length: Frames per block.
        frame_length: Samples per frame.
        hop_length: Samples between frames.
        sr: Resample to this rate. Default: keep the file's rate.
        mono: Mix to one channel.
        offset: Start in seconds.
        duration: Length in seconds. Default: to the end.
        fill_value: Pad the last block to full length with this value; otherwise a short last
            block is returned as it is.
        res_type: Resampler quality when ``sr`` is given.

    Yields:
        Arrays with time on the last axis.

    Raises:
        ValueError: A parameter is not positive.
    """
    for name, value in (("block_length", block_length), ("frame_length", frame_length), ("hop_length", hop_length)):
        if value <= 0:
            raise ValueError(f"{name}: must be positive")
    block = frame_length + (block_length - 1) * hop_length
    step = block_length * hop_length

    def finished(part: NDArray[Any], last: bool) -> NDArray[Any]:
        if last and fill_value is not None and part.shape[-1] < block:
            width = [(0, 0)] * (part.ndim - 1) + [(0, block - part.shape[-1])]
            return np.pad(part, width, constant_values=fill_value)
        return part

    file = os.fspath(path)
    if sr is not None and sr != _aus.io.info(file).sample_rate:
        audio = read_segment(file, offset=offset, duration=duration)
        audio = audio.resample(sr, res_type if res_type in ("fast", "medium", "high") else "high")
        data = audio.to_numpy()
        if mono and data.ndim > 1:
            data = data.mean(axis=0)
        for start in range(0, data.shape[-1], step):
            last = start + block >= data.shape[-1]
            yield finished(data[..., start : start + block], last)
            if last:
                return
        return
    # The file's rate: each block reads only the samples it does not have yet.
    reader = _aus.io.StreamedAudioReader(file)
    try:
        rate = reader.sample_rate
        first = int(offset * rate)
        if first >= reader.total_frames:
            return
        reader.seek_to_frame(first)
        left = reader.total_frames - first
        if duration is not None:
            left = min(left, int(duration * rate))
        dtype = _native_type(file)
        kept: NDArray[Any] | None = None
        while left > 0:
            have = 0 if kept is None else kept.shape[-1]
            count = min(block - have, left)
            chunk = reader.read_frames(count, dtype) if count > 0 else None
            if chunk is not None:
                fresh = chunk.to_numpy()
                left -= fresh.shape[-1]
                kept = fresh if kept is None else np.concatenate([kept, fresh], axis=-1)
            elif count > 0:
                left = 0
            if kept is None:
                return
            part = kept.mean(axis=0) if mono and kept.ndim > 1 else kept
            yield finished(part, left <= 0)
            if left <= 0:
                return
            kept = kept[..., step:] if step < kept.shape[-1] else None
            if step > block:
                skip = min(step - block, left)
                left -= skip
                reader.seek_to_frame(reader.current_frame + skip)
    finally:
        reader.close()


def _from_rows(y: NDArray[Any], sr: int) -> Any:
    return _aus.AudioSamples.new_mono(y, sr) if y.ndim == 1 else _aus.AudioSamples.new_multi(y, sr)


def resample(
    y: NDArray[Any],
    *,
    orig_sr: int,
    target_sr: int,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
) -> NDArray[Any]:
    """Resamples an array.

    Args:
        y: Samples with time on the last axis, 1-D or 2-D.
        orig_sr: Its rate in Hz.
        target_sr: The new rate in Hz.
        res_type: ``"soxr_hq"``, ``"soxr_mq"``, ``"soxr_lq"``, ``"high"``, ``"medium"`` or ``"fast"``.
        fix: The length is ``ceil(n * target_sr / orig_sr)``. The resampler always gives this
            length, so this has no other value.
        scale: Divide by the square root of the ratio, which keeps total energy (as one
            multiplication by its reciprocal: within one rounding of the quotient).

    Returns:
        The resampled array, of ``y``'s dtype.
    """
    quality = {"soxr_hq": "high", "soxr_vhq": "high", "soxr_mq": "medium", "soxr_lq": "fast", "soxr_qq": "fast"}.get(res_type, res_type)
    out = _from_rows(np.asarray(y), orig_sr)._resample_array(target_sr, quality)
    if scale:
        # A multiplication is several times cheaper than a division per sample.
        out *= 1.0 / np.sqrt(target_sr / orig_sr)
    return out


def pitch_shift(y: NDArray[Any], *, sr: int, n_steps: float, bins_per_octave: int = 12, scale: bool = False, **kwargs: Any) -> NDArray[Any]:
    """Shifts the pitch of an array without changing its length.

    Args:
        y: Samples with time on the last axis, 1-D or 2-D.
        sr: Its rate in Hz.
        n_steps: Fractional steps to shift by.
        bins_per_octave: Steps per octave.
        scale: Divide by the square root of the resampling ratio, as :func:`resample`.
        **kwargs: Passed to ``AudioSamples.pitch_shift``.

    Returns:
        The shifted array.
    """
    audio = _from_rows(np.asarray(y), sr)
    out = audio.pitch_shift(n_steps, bins_per_octave=bins_per_octave, **kwargs).to_numpy()
    if scale:
        out *= 1.0 / np.sqrt(2.0 ** (-float(n_steps) / bins_per_octave))
    return out


def frame(x: NDArray[Any], *, frame_length: int, hop_length: int, axis: int = -1, writeable: bool = False, subok: bool = False) -> NDArray[Any]:
    """Slices an array into overlapping frames along any axis.

    Args:
        x: The array.
        frame_length: Samples per frame.
        hop_length: Samples between frames.
        axis: The axis to frame. With a negative axis the frames take the last position; with
            a positive axis they take its place.
        writeable: The result may be written to, which writes through to ``x``.
        subok: Keep an array subclass.

    Returns:
        A view with one more dimension.

    Raises:
        ValueError: The axis is shorter than one frame, or a length is not positive.
    """
    array = np.asanyarray(x) if subok else np.asarray(x)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length, hop_length: must be positive")
    if array.shape[axis] < frame_length:
        raise ValueError(f"x: {array.shape[axis]} samples along axis {axis} are fewer than a frame")
    frames = 1 + (array.shape[axis] - frame_length) // hop_length
    position = axis if axis >= 0 else array.ndim + axis
    stride = array.strides[position]
    if axis >= 0:
        shape = (*array.shape[:position], frames, frame_length, *array.shape[position + 1 :])
        strides = (*array.strides[:position], stride * hop_length, stride, *array.strides[position + 1 :])
    else:
        shape = (*array.shape[:position], frame_length, frames, *array.shape[position + 1 :])
        strides = (*array.strides[:position], stride, stride * hop_length, *array.strides[position + 1 :])
    return np.lib.stride_tricks.as_strided(array, shape, strides, subok, writeable)


def _mask(events: NDArray[Any], n: int) -> NDArray[np.bool_]:
    out = np.zeros(n, dtype=bool)
    out[np.asarray(events, dtype=np.int64)] = True
    return out


def _envelope(y: NDArray[Any] | None, sr: float, onset_envelope: NDArray[Any] | None, hop_length: int) -> NDArray[Any]:
    if onset_envelope is not None:
        return onset_envelope
    if y is None:
        raise ValueError("y: give y or onset_envelope")
    return np.asarray(_from_rows(np.asarray(y), int(sr)).onset_strength(hop_length=hop_length))[0]


def onset_detect(*, y: NDArray[Any] | None = None, sr: float = 22050, onset_envelope: NDArray[Any] | None = None, hop_length: int = 512, sparse: bool = True, **kwargs: Any) -> NDArray[Any]:
    """Onset frames, as indices or as a boolean mask over the frames.

    Args:
        y: Samples, when ``onset_envelope`` is not given.
        sr: Sample rate in Hz.
        onset_envelope: Onset strength per frame.
        hop_length: Samples between frames.
        sparse: Return indices; otherwise a boolean mask.
        **kwargs: Passed to ``rhythm.onset_detect``.

    Returns:
        Indices, or a mask of one element per frame.
    """
    from audio_samples import rhythm

    env = _envelope(y, sr, onset_envelope, hop_length)
    events = rhythm.onset_detect(onset_envelope=env, sr=sr, hop_length=hop_length, **kwargs)
    return events if sparse else _mask(events, np.shape(env)[-1])


def beat_track(*, y: NDArray[Any] | None = None, sr: float = 22050, onset_envelope: NDArray[Any] | None = None, hop_length: int = 512, sparse: bool = True, **kwargs: Any) -> tuple[Any, NDArray[Any]]:
    """Tempo and beat frames, as indices or as a boolean mask over the frames.

    Args:
        y: Samples, when ``onset_envelope`` is not given.
        sr: Sample rate in Hz.
        onset_envelope: Onset strength per frame.
        hop_length: Samples between frames.
        sparse: Return indices; otherwise a boolean mask.
        **kwargs: Passed to ``rhythm.beat_track``.

    Returns:
        The tempo, and the indices or a mask of one element per frame.
    """
    from audio_samples import rhythm

    env = _envelope(y, sr, onset_envelope, hop_length)
    tempo, beats = rhythm.beat_track(onset_envelope=env, sr=sr, hop_length=hop_length, **kwargs)
    return tempo, beats if sparse else _mask(beats, np.shape(env)[-1])


def _normalize(x: NDArray[Any], norm: float | None) -> NDArray[Any]:
    """Normalises along the second-to-last axis; all-zero frames stay zero."""
    if norm is None:
        return x
    magnitudes = np.abs(x)
    if np.isinf(norm):
        lengths = magnitudes.max(axis=-2, keepdims=True)
    else:
        lengths = np.sum(magnitudes**norm, axis=-2, keepdims=True) ** (1.0 / norm)
    small = np.finfo(np.float32 if x.dtype == np.float32 else np.float64).tiny
    return np.where(lengths < small, 0.0, x / np.where(lengths < small, 1.0, lengths))


def stft(
    y: NDArray[Any],
    *,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Any = "hann",
    center: bool = True,
    pad_mode: Any = "constant",
    sr: int = 22050,
    dtype: Any = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Short-time Fourier transform of an array, optionally into a given buffer.

    Args:
        y: Samples with time on the last axis, 1-D or 2-D.
        n_fft, hop_length, win_length, window, center, pad_mode: As ``AudioSamples.stft_matrix``.
        sr: Rate of the samples; the transform does not depend on it.
        dtype: ``complex64`` or ``complex128``, by converting the samples first. None follows
            the samples.
        out: Array to write the result to. It may be longer in time than the result; the
            frames that are written are returned as a view of it.

    Returns:
        The complex spectrogram.
    """
    data = np.asarray(y)
    if dtype is not None:
        data = data.astype(np.float32 if np.dtype(dtype) == np.complex64 else np.float64)
    audio = _from_rows(data, sr)
    # A buffer of exactly the type of the result is written in place.
    matching = {np.dtype(np.float32): np.complex64, np.dtype(np.float64): np.complex128}.get(data.dtype)
    if isinstance(out, np.ndarray) and out.dtype == matching and out.flags.writeable and out.ndim == data.ndim + 1:
        frames = audio._stft_matrix_into(
            out, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode
        )
        return out[..., :frames]
    result = audio.stft_matrix(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode
    )
    if data.ndim == 1 and result.ndim == 3:
        result = result[0]
    if out is None:
        return result
    if out.shape[:-1] != result.shape[:-1] or out.shape[-1] < result.shape[-1]:
        raise ValueError(f"out: has shape {out.shape}, expected at least {result.shape}")
    view = out[..., : result.shape[-1]]
    view[...] = result
    return view


def istft(
    stft_matrix: NDArray[Any],
    *,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: Any = "hann",
    center: bool = True,
    length: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Inverse short-time Fourier transform, optionally into a given buffer.

    Args:
        stft_matrix: The complex spectrogram.
        hop_length, win_length, n_fft, window, center, length: As ``spectral.istft``.
        out: Array to write the result to, of at least the result's length. The samples that
            are written are returned as a view of it.

    Returns:
        The samples.
    """
    result = _aus.spectral.istft(
        stft_matrix, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window=window, center=center, length=length
    )
    if out is None:
        return result
    if out.shape[:-1] != result.shape[:-1] or out.shape[-1] < result.shape[-1]:
        raise ValueError(f"out: has shape {out.shape}, expected at least {result.shape}")
    view = out[..., : result.shape[-1]]
    view[...] = result
    return view


def reassigned_spectrogram(
    y: NDArray[Any],
    *,
    sr: int = 22050,
    S: NDArray[Any] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Any = "hann",
    center: bool = True,
    reassign_frequencies: bool = True,
    reassign_times: bool = True,
    ref_power: float = 1e-6,
    fill_nan: bool = False,
    clip: bool = True,
    pad_mode: Any = "constant",
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Reassigned spectrogram with a given spectrogram for the main window.

    Without ``S`` this is ``AudioSamples.reassigned_spectrogram``. With ``S``, its values
    replace those of the STFT with the analysis window, in the corrections and in the
    magnitudes; the two other STFTs still come from ``y``.

    Args:
        y: Samples with time on the last axis.
        sr: Rate of the samples in Hz.
        S: Complex spectrogram of the analysis window.
        n_fft, hop_length, win_length, window, center, pad_mode: As for the STFT.
        reassign_frequencies: Correct the frequencies.
        reassign_times: Correct the times.
        ref_power: Bins below this power, relative to a callable's result if it is one, have no
            correction.
        fill_nan: Fill uncorrected bins with the bin centres; otherwise NaN.
        clip: Keep corrections within the frequency and time range of the spectrogram.

    Returns:
        ``(frequencies, times, magnitudes)``.
    """
    if n_fft < 2:
        raise ValueError(f"n_fft: must be at least 2, got {n_fft}")
    data = np.asarray(y)
    hop = hop_length if hop_length is not None else (win_length or n_fft) // 4
    length = win_length or n_fft
    base = np.asarray(_aus.spectral.get_window(window, length, fftbins=True), dtype=np.float64)
    pad = n_fft - length
    base = np.pad(base, (pad // 2, pad - pad // 2))
    audio = _from_rows(data, sr)

    def transform(w: NDArray[Any]) -> NDArray[Any]:
        kept = audio.stft_matrix(n_fft=n_fft, hop_length=hop, window=w, center=center, pad_mode=pad_mode)
        return kept[0] if data.ndim == 1 and kept.ndim == 3 else kept

    S_h = transform(base) if S is None else np.asarray(S)
    half = n_fft // 2
    bins = np.linspace(0.0, sr / 2.0, half + 1)
    bins = bins.reshape((-1, 1))
    frame_times = np.arange(S_h.shape[-1]) * hop / sr
    if not center:
        frame_times = frame_times + (n_fft // 2) / sr
    mags = np.abs(S_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        if reassign_frequencies:
            derivative = 0.5 * (np.roll(base, -1) - np.roll(base, 1))
            freqs = bins + (-np.imag(transform(derivative) / S_h)) * (0.5 * sr / np.pi)
        else:
            freqs = np.broadcast_to(bins, mags.shape).copy()
        if reassign_times:
            positions = np.arange(-half, half + 1) if n_fft % 2 else np.arange(0.5 - half, half)
            times = frame_times + np.real(transform(base * positions) / S_h) / sr
        else:
            times = np.broadcast_to(frame_times, mags.shape).copy()
    power = mags * mags
    power_threshold = ref_power(power) if callable(ref_power) else ref_power
    unset = power < power_threshold
    del power
    freqs[unset] = np.nan
    times[unset] = np.nan
    if fill_nan:
        missing = np.isnan(freqs)
        freqs[missing] = np.broadcast_to(bins, freqs.shape)[missing]
        missing = np.isnan(times)
        times[missing] = np.broadcast_to(frame_times, times.shape)[missing]
    if clip:
        np.clip(freqs, 0.0, sr / 2.0, out=freqs)
        np.clip(times, 0.0, data.shape[-1] / sr, out=times)
    return freqs, times, mags


def chroma_cqt(
    *,
    C: NDArray[Any],
    fmin: float | None = None,
    norm: float | None = np.inf,
    threshold: float | None = 0.0,
    n_chroma: int = 12,
    window: NDArray[Any] | None = None,
    bins_per_octave: int = 36,
) -> NDArray[Any]:
    """Chromagram of a constant-Q magnitude spectrogram.

    Args:
        C: Magnitudes, ``(..., n_bins, frames)``.
        fmin: Frequency of the first bin in Hz. None means C1.
        norm: Normalisation of each frame.
        threshold: Values below this become zero.
        n_chroma: Chroma bins.
        window: Smoothing weights for each chroma row across constant-Q bins.
        bins_per_octave: Constant-Q bins per octave.

    Returns:
        The chromagram, in the precision of ``C``.
    """
    C = np.asarray(C)
    basis = _aus.features.cq_to_chroma(
        C.shape[-2], bins_per_octave=bins_per_octave, n_chroma=n_chroma, fmin=fmin, window=window
    )
    chroma = basis.astype(C.dtype, copy=False) @ C
    del basis
    if threshold is not None:
        chroma[chroma < threshold] = 0.0
    return _normalize(chroma, norm)


def chroma_cens(
    *,
    C: NDArray[Any],
    fmin: float | None = None,
    n_chroma: int = 12,
    bins_per_octave: int = 36,
    window: NDArray[Any] | None = None,
    norm: float | None = 2,
    win_len_smooth: int | None = 41,
    smoothing_window: Any = "hann",
) -> NDArray[Any]:
    """Chroma energy normalised statistics of a constant-Q magnitude spectrogram.

    Args:
        C: Magnitudes, ``(..., n_bins, frames)``.
        fmin, n_chroma, bins_per_octave, window: As :func:`chroma_cqt`.
        norm: Final normalisation of each frame.
        win_len_smooth: Length of the time smoothing; None for none.
        smoothing_window: Window of the time smoothing.

    Returns:
        The features.
    """
    chroma = chroma_cqt(C=C, fmin=fmin, n_chroma=n_chroma, bins_per_octave=bins_per_octave, window=window, norm=None)
    return _aus.features.chroma_cens_from_chroma(  # type: ignore[no-any-return]
        chroma, norm=norm, win_len_smooth=win_len_smooth, smoothing_window=smoothing_window
    )


def chroma_vqt(
    *,
    V: NDArray[Any],
    fmin: float | None = None,
    norm: float | None = np.inf,
    threshold: float | None = 0.0,
    bins_per_octave: int = 12,
) -> NDArray[Any]:
    """Chromagram with one bin per variable-Q bin of an octave.

    Args:
        V: Magnitudes, ``(..., n_bins, frames)``.
        fmin: Frequency of the first bin in Hz. None means C1.
        norm: Normalisation of each frame.
        threshold: Values below this become zero.
        bins_per_octave: Variable-Q bins per octave, and chroma bins.

    Returns:
        The chromagram.
    """
    return chroma_cqt(C=V, fmin=fmin, norm=norm, threshold=threshold, n_chroma=bins_per_octave, bins_per_octave=bins_per_octave)


def tempogram_ratio(
    *,
    tg: NDArray[Any],
    bpm: NDArray[Any],
    sr: float = 22050,
    hop_length: int = 512,
    freqs: NDArray[Any] | None = None,
    factors: NDArray[Any] | None = None,
    kind: Any = "linear",
    fill_value: float = 0.0,
) -> NDArray[Any]:
    """Tempogram at multiples of the tempo, with given tempo frequencies.

    Args:
        tg: Tempogram, ``(..., n_lags, frames)``.
        bpm: Tempo per frame, in beats per minute.
        sr, hop_length: Used when ``freqs`` is None.
        freqs: The tempo of each row of ``tg``.
        factors: Multiples of the tempo. None means the metric multiples of
            Prockup et al. (2015).
        kind: Interpolation kind.
        fill_value: Value outside the tempo range.

    Returns:
        ``tg`` interpolated at ``factors * bpm``, ``(..., n_factors, frames)``.
    """
    if freqs is None:
        freqs = _aus.rhythm.tempo_frequencies(tg.shape[-2], sr=sr, hop_length=hop_length)
    if factors is None:
        factors = np.array([4, 8 / 3, 3, 2, 4 / 3, 3 / 2, 1, 2 / 3, 3 / 4, 1 / 2, 1 / 3, 3 / 8, 1 / 4])
    return _aus.features.f0_harmonics(  # type: ignore[no-any-return]
        tg, freqs=freqs, f0=bpm, harmonics=factors, kind=kind, fill_value=fill_value
    )


def onset_strength(
    *,
    y: NDArray[Any] | None = None,
    sr: float = 22050,
    S: NDArray[Any] | None = None,
    lag: int = 1,
    max_size: int = 1,
    ref: NDArray[Any] | None = None,
    detrend: bool = False,
    center: bool = True,
    feature: Any = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs: Any,
) -> NDArray[Any]:
    """Onset strength with any time-frequency feature.

    Args:
        y: Samples, if ``S`` is None.
        sr: Rate of the samples in Hz.
        S: Log-power spectrogram, ``(..., n_bins, frames)``.
        lag: Frames between the compared spectra.
        max_size: Size of the frequency maximum filter on the reference.
        ref: Reference spectrogram, of the shape of ``S``.
        detrend: Remove the slowly varying component.
        center: Align the envelope with centred frames.
        feature: Called as ``feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs)``;
            its magnitude in decibels, limited to 80 dB below the peak, is the spectrogram. None uses the mel spectrogram.
        n_fft: Passed to ``feature``, and used by ``center``.
        hop_length: Passed to ``feature``, and used by ``center``.
        **kwargs: Passed to ``feature``.

    Returns:
        The envelope of each channel, with one row removed for a single channel.
    """
    if S is None:
        if y is None:
            raise ValueError("y: give y or S")
        if feature is None:
            audio = _from_rows(np.asarray(y), int(sr))
            power = np.abs(audio.melspectrogram(n_fft=n_fft, hop_length=hop_length, **kwargs))
        else:
            power = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
        S = _aus.spectral.power_to_db(power, top_db=80.0)
        del power
    return _aus.rhythm.onset_strength(  # type: ignore[no-any-return]
        S=np.asarray(S), sr=int(sr), lag=lag, max_size=max_size, ref=ref, detrend=detrend, center=center,
        n_fft=n_fft, hop_length=hop_length,
    )


def salience(
    S: NDArray[Any],
    *,
    freqs: NDArray[Any],
    harmonics: Any,
    weights: Any = None,
    aggregate: Any = None,
    filter_peaks: bool = True,
    fill_value: float = np.nan,
    kind: Any = "linear",
    axis: int = -2,
) -> NDArray[Any]:
    """Harmonic salience with any interpolant along any frequency axis.

    Args:
        S: Energy.
        freqs: Frequencies of the frequency axis, 1-D.
        harmonics: Harmonic multipliers.
        weights: Weight of each harmonic in the default aggregate. None weighs them equally.
        aggregate: Called as ``aggregate(values, axis=...)``. None uses the weighted average.
        filter_peaks: Keep only the local maxima of ``S`` along the frequency axis.
        fill_value: Value everywhere else.
        kind: Interpolation kind.
        axis: The frequency axis.

    Returns:
        The salience, of the shape of ``S``.
    """
    S = np.asarray(S)
    stacked = _aus.features.interp_harmonics(S, freqs=freqs, harmonics=harmonics, kind=kind, axis=axis)
    harmonic_axis = (axis if axis < 0 else axis - S.ndim) - 1
    if aggregate is None:
        factors = np.ones(len(harmonics)) if weights is None else np.asarray(weights, dtype=np.float64)
        result = np.average(stacked, axis=harmonic_axis, weights=factors)
    else:
        result = aggregate(stacked, axis=harmonic_axis)
    if not filter_peaks:
        return result  # type: ignore[no-any-return]
    moved = np.moveaxis(S, axis, -1)
    peaks = np.zeros(moved.shape, dtype=bool)
    peaks[..., 1:-1] = (moved[..., 1:-1] > moved[..., :-2]) & (moved[..., 1:-1] > moved[..., 2:])
    peaks = np.moveaxis(peaks, -1, axis)
    return np.where(peaks, result, fill_value)
