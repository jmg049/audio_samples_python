import os
import random
import soundfile as sf
import audio_python as aus
from argparse import ArgumentParser
from functools import partial


ITERS = 2000  # increase if the flamegraph is too thin
DTYPE = "int16"

def read_many(fn, path, iters):
    """
    Execute the hot path many times in one Python call.

    This maximises native execution time per sample and
    produces clean, meaningful flamegraphs.
    """
    for _ in range(iters):
        fn(path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--iters", type=int, default=ITERS,
        help="Number of iterations to run for each reader"
    )
    parser.add_argument("--dtype", type=str, default=DTYPE, help="Data type to read as")
    parser.add_argument("--method", type=str, choices=["rust", "python"], default="rust",)
    args = parser.parse_args()
    ITERS = args.iters
    DTYPE = args.dtype
    random.seed(42)
    input_files = [
        os.path.join("/home/jmg/Music/Deadbeat/", f)
        for f in os.listdir("/home/jmg/Music/Deadbeat/")
        if f.endswith(".wav")
    ]
    print(f"Found {len(input_files)} input files.")
    print("Sample input files:", input_files[:5])

    read_rust = aus.io.read_as_i16
    read_py = partial(sf.read, dtype=DTYPE)
    METHOD_FN = read_rust if args.method == "rust" else read_py

    print(f"Running {args.method} reader for {ITERS} iterations as {DTYPE}.")


    INPUT_FILE = "./test_output.wav" 
    # input_files[
    #     input_files.index("/home/jmg/Music/Deadbeat/09 - Ethereal Connection.flac.wav")
    # ]
    print(f"Using input file: {INPUT_FILE}")
    read_many(METHOD_FN, INPUT_FILE, ITERS)
    print("Done.")