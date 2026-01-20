File I/O Operations
===================
.. automodule:: audio_samples.io
   :members:

.. currentmodule:: audio_samples.io

The io module provides high-performance audio file reading and writing capabilities. All operations
preserve audio properties automatically and support multiple audio formats with type-safe conversions.

Reading Audio Files
-------------------

.. autofunction:: read

.. autofunction:: read_with_info

Writing Audio Files
-------------------

.. autofunction:: save

.. autofunction:: save_as_type

File Information
----------------

.. autoclass:: AudioInfo
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic File Operations
~~~~~~~~~~~~~~~~~~~~~

Read and write audio files with automatic format detection:

.. code-block:: python

   import audio_samples as aus

   # Read audio file (format detected from extension)
   audio = aus.io.read("input.wav")

   # Audio properties are preserved automatically
   print(f"Loaded: {audio.num_channels()} channel(s), {audio.sample_rate()} Hz")
   print(f"Duration: {audio.duration_seconds():.2f} seconds")

   # Process the audio
   audio.scale(0.5)  # Reduce volume by 50%
   audio.fade_in(0.1, 'exponential')  # 100ms fade-in

   # Save with preserved format
   aus.io.save("output.wav", audio)

Reading with Type Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control the data type during loading:

.. code-block:: python

   import numpy as np

   # Read as specific type
   audio_f32 = aus.io.read("audio.wav", as_type=np.float32)
   audio_i16 = aus.io.read("audio.wav", as_type=np.int16)

   # Check the resulting types
   print(f"f32 audio dtype: {audio_f32.dtype}")
   print(f"i16 audio dtype: {audio_i16.dtype}")

File Information Inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get detailed information about audio files before loading:

.. code-block:: python

   # Read with metadata
   audio, info = aus.io.read_with_info("large_file.wav")

   print(f"File info:")
   print(f"  Sample rate: {info.sample_rate} Hz")
   print(f"  Channels: {info.channels}")
   print(f"  Bits per sample: {info.bits_per_sample}")
   print(f"  Duration: {info.duration:.2f} seconds")
   print(f"  Total samples per channel: {info.num_samples}")
   print(f"  Sample type: {info.sample_type}")

Format Conversion During Save
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert audio format when saving:

.. code-block:: python

   # Load audio (might be any format)
   audio = aus.io.read("input.wav")

   # Save as different formats
   aus.io.save_as_type("output_16bit.wav", audio, np.int16)
   aus.io.save_as_type("output_32bit.wav", audio, np.int32)
   aus.io.save_as_type("output_float.wav", audio, np.float32)

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files efficiently:

.. code-block:: python

   import pathlib

   input_dir = pathlib.Path("input_files")
   output_dir = pathlib.Path("processed_files")
   output_dir.mkdir(exist_ok=True)

   for wav_file in input_dir.glob("*.wav"):
       # Read audio
       audio = aus.io.read(str(wav_file))

       # Apply consistent processing
       audio.normalize(-1.0, 1.0, 'peak')
       audio.remove_dc_offset()

       # Save to output directory
       output_path = output_dir / wav_file.name
       aus.io.save(str(output_path), audio)

       print(f"Processed {wav_file.name}")

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

AudioSamples I/O is optimized for performance:

.. code-block:: python

   import time

   # Time file operations
   start_time = time.time()
   large_audio = aus.io.read("large_file.wav")
   read_time = time.time() - start_time

   print(f"Read {large_audio.duration_seconds():.1f}s of audio in {read_time:.4f}s")
   print(f"Real-time factor: {large_audio.duration_seconds() / read_time:.1f}x")

   # Save operations are equally fast
   start_time = time.time()
   aus.io.save("output_large.wav", large_audio)
   write_time = time.time() - start_time

   print(f"Wrote audio in {write_time:.4f}s")
   print(f"Real-time factor: {large_audio.duration_seconds() / write_time:.1f}x")