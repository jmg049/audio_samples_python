"""
End-to-end DTMF demonstration using audio_samples.

This script illustrates:
- Programmatic DTMF tone synthesis
- WAV serialisation via audio_samples
- Short-time Fourier analysis
- Frequency-based DTMF detection
- Temporal debouncing of per-frame detections

"""

import time
from argparse import ArgumentParser

import audio_samples as aus
from dtmf_table import DtmfKey, DtmfTable


DTMF_TABLE = DtmfTable()


def dtmf_keys_from_sequence(sequence: str) -> list[DtmfKey]:
    """
    Convert a character sequence into DTMF key symbols.

    Parameters
    ----------
    sequence : str
        String of valid DTMF characters.

    Returns
    -------
    list[DtmfKey]
    """
    return [DtmfKey.from_char(symbol) for symbol in sequence]


def generate_dtmf_tone(
    keys: list[DtmfKey],
    duration_per_symbol_secs: float,
    sample_rate: int,
    amplitude: float,
) -> aus.AudioSamples:
    """
    Generate a DTMF signal for a sequence of keys.

    Each symbol is generated as the sum of its low and high
    sinusoidal components, separated by short silence gaps.
    """
    silence_duration_secs = 0.02
    silence = aus.generation.silence(
        duration_secs=silence_duration_secs,
        sample_rate=sample_rate,
    )

    segments: list[aus.AudioSamples] = []

    for i, key in enumerate(keys):
        low_freq, high_freq = DTMF_TABLE.lookup_key(key)

        low_tone = aus.generation.sine_wave(
            frequency=low_freq,
            duration_secs=duration_per_symbol_secs,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )
        high_tone = aus.generation.sine_wave(
            frequency=high_freq,
            duration_secs=duration_per_symbol_secs,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )

        # Average to avoid doubling peak amplitude
        tone = (low_tone + high_tone)
        tone.scale(0.5)
        segments.append(tone)

        if i < len(keys) - 1:
            segments.append(silence)

    return aus.AudioSamples.concatenate(segments)


def debounce_frames(
    frame_keys: list[DtmfKey | None],
    min_len: int,
    max_gap: int,
    max_force_len: int,
) -> list[DtmfKey]:
    """
    Convert per-frame DTMF detections into a stable key sequence.

    A key is emitted if it is detected for at least `min_len`
    consecutive frames. Short gaps up to `max_gap` frames are
    tolerated. Sustained detections longer than `max_force_len`
    are forcibly split.
    """
    decoded: list[DtmfKey] = []

    current_key: DtmfKey | None = None
    current_len = 0
    gap_len = 0

    for frame_key in frame_keys:
        # Forced split of overly long detections
        if frame_key is not None and current_len >= max_force_len:
            decoded.append(current_key)
            current_key = None
            current_len = 0
            gap_len = 0

        if current_key is not None and frame_key == current_key:
            current_len += 1
            gap_len = 0

        elif current_key is not None and frame_key is None:
            gap_len += 1
            if gap_len > max_gap:
                if current_len >= min_len:
                    decoded.append(current_key)
                current_key = None
                current_len = 0
                gap_len = 0

        elif current_key is not None and frame_key is not None:
            if current_len >= min_len:
                decoded.append(current_key)
            current_key = frame_key
            current_len = 1
            gap_len = 0

        elif current_key is None and frame_key is not None:
            current_key = frame_key
            current_len = 1
            gap_len = 0

    if current_key is not None and current_len >= min_len:
        decoded.append(current_key)

    return decoded


def detect_dtmf_sequence(
    signal: aus.AudioSamples,
    window_size_secs: float,
    min_len: int,
    max_gap: int,
    max_force_len: int,
    tolerance_hz: float = 50.0,
) -> list[DtmfKey]:
    """
    Detect a DTMF key sequence from an audio signal.
    """
    sample_rate = signal.sample_rate
    window_size = int(window_size_secs * sample_rate)
    hop_size = window_size // 2

    stft_matrix = signal.stft(
        window_size=window_size,
        hop_size=hop_size,
        window_type="hann",
    )

    num_freq_bins = window_size // 2 + 1
    freq_resolution = sample_rate / window_size

    def bin_to_freq(bin_index: int) -> float:
        return bin_index * freq_resolution

    low_bin_start = max(0, int(650.0 / freq_resolution))
    low_bin_end = min(num_freq_bins - 1, int(1000.0 / freq_resolution))
    high_bin_start = max(0, int(1150.0 / freq_resolution))
    high_bin_end = min(num_freq_bins - 1, int(1700.0 / freq_resolution))

    detected_per_frame: list[DtmfKey | None] = []

    for frame in stft_matrix.T:
        best_low_bin: int | None = None
        best_low_mag = 0.0

        for bin_index in range(low_bin_start, low_bin_end + 1):
            magnitude = abs(frame[bin_index])
            if magnitude > best_low_mag:
                best_low_mag = magnitude
                best_low_bin = bin_index

        best_high_bin: int | None = None
        best_high_mag = 0.0

        for bin_index in range(high_bin_start, high_bin_end + 1):
            magnitude = abs(frame[bin_index])
            if magnitude > best_high_mag:
                best_high_mag = magnitude
                best_high_bin = bin_index

        if best_low_bin is None or best_high_bin is None:
            detected_per_frame.append(None)
            continue

        low_freq_hz = bin_to_freq(best_low_bin)
        high_freq_hz = bin_to_freq(best_high_bin)

        detected_key = DTMF_TABLE.from_pair_tol_f64(
            low_freq_hz,
            high_freq_hz,
            tolerance_hz,
        )
        detected_per_frame.append(detected_key)

    # Pad with silence to flush final active detections
    detected_per_frame.extend([None] * (min_len + max_gap + 1))

    return debounce_frames(
        detected_per_frame,
        min_len=min_len,
        max_gap=max_gap,
        max_force_len=max_force_len,
    )


def main(args) -> None:
    sequence = args.sequence
    keys = dtmf_keys_from_sequence(sequence)

    tone = generate_dtmf_tone(
        keys=keys,
        duration_per_symbol_secs=args.duration_per_symbol_secs,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
    )

    safe_sequence = sequence.replace("*", "star").replace("#", "hash")
    out_file_name = f"dtmf_{safe_sequence}.wav"

    aus.io.save(out_file_name, tone)

    start_time = time.time()
    detected_sequence = detect_dtmf_sequence(
        tone,
        window_size_secs=args.window_size_secs,
        min_len=args.min_len,
        max_gap=args.max_gap,
        max_force_len=args.max_force_len,
    )
    elapsed_time = time.time() - start_time

    decoded = "".join(key.to_char() for key in detected_sequence)

    print(f"Input sequence   : {sequence}")
    print(f"Detected sequence: {decoded}")
    print(f"Processing time  : {elapsed_time:.4f} s")


if __name__ == "__main__":
    parser = ArgumentParser(description="DTMF generation and detection demo")
    parser.add_argument(
        "sequence",
        type=str,
        help="DTMF sequence (digits, *, #)",
    )
    parser.add_argument(
        "--duration-per-symbol-secs",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--window-size-secs",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-force-len",
        type=int,
        default=10,
    )

    main(parser.parse_args())
