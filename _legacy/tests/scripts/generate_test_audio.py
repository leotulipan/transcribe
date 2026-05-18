"""
Generate test audio files for testing.

Usage:
    uv run python tests/scripts/generate_test_audio.py
"""
import struct
import subprocess
from pathlib import Path


def generate_silent_wav(duration_seconds: int, output_path: Path) -> None:
    """
    Generate silent WAV file using Python stdlib (no ffmpeg needed).

    Args:
        duration_seconds: Duration of silence in seconds
        output_path: Where to save the WAV file
    """
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    num_samples = duration_seconds * sample_rate

    with open(output_path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + num_samples * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8))
        f.write(struct.pack('<H', num_channels * bits_per_sample // 8))
        f.write(struct.pack('<H', bits_per_sample))
        f.write(b'data')
        f.write(struct.pack('<I', num_samples * 2))
        # Silent data (zeros)
        f.write(b'\x00\x00' * num_samples)


def convert_with_ffmpeg(input_path: Path, output_path: Path, bitrate: str = None) -> None:
    """
    Convert audio using ffmpeg.

    Args:
        input_path: Source audio file
        output_path: Destination path
        bitrate: Optional bitrate (e.g., "128k")
    """
    cmd = ['ffmpeg', '-y', '-i', str(input_path), '-ac', '1', '-ar', '16000']
    if bitrate:
        cmd.extend(['-b:a', bitrate])
    cmd.append(str(output_path))

    subprocess.run(cmd, check=True, capture_output=True)


def convert_to_flac(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    """
    Convert audio to FLAC format.

    Args:
        input_path: Source audio file
        output_path: Destination FLAC path
        sample_rate: Target sample rate in Hz
    """
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_path),
        '-ac', '1',  # mono
        '-ar', str(sample_rate),  # sample rate
        str(output_path)
    ], check=True, capture_output=True)


def generate_stereo_wav(duration_seconds: int, output_path: Path) -> None:
    """
    Generate stereo WAV file (for validation testing).

    Args:
        duration_seconds: Duration of silence in seconds
        output_path: Where to save the WAV file
    """
    sample_rate = 16000
    num_channels = 2  # stereo
    bits_per_sample = 16
    num_samples = duration_seconds * sample_rate

    with open(output_path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + num_samples * num_channels * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8))
        f.write(struct.pack('<H', num_channels * bits_per_sample // 8))
        f.write(struct.pack('<H', bits_per_sample))
        f.write(b'data')
        f.write(struct.pack('<I', num_samples * num_channels * 2))
        # Silent data (zeros, stereo)
        f.write(b'\x00\x00\x00\x00' * num_samples)


def main():
    """Generate all test audio files."""
    fixtures_dir = Path(__file__).parent.parent / 'fixtures' / 'audio_files'
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Source file (existing)
    source = fixtures_dir / 'sample_speech.m4a'

    print(f"Generating test audio files in: {fixtures_dir}")

    # 1. Generate silent WAV files (no dependencies)
    print("Generating silent WAV files...")
    generate_silent_wav(10, fixtures_dir / 'silent_10s.wav')
    print("  - silent_10s.wav")

    generate_silent_wav(30, fixtures_dir / 'silent_30s.wav')
    print("  - silent_30s.wav")

    generate_silent_wav(60, fixtures_dir / 'silent_60s.wav')
    print("  - silent_60s.wav")

    generate_silent_wav(600, fixtures_dir / 'silent_600s.wav')
    print("  - silent_600s.wav (10 min for chunking tests)")

    # 2. Generate stereo WAV (for format validation tests)
    generate_stereo_wav(5, fixtures_dir / 'stereo_5s.wav')
    print("  - stereo_5s.wav (for validation testing)")

    # 3. Convert existing file to other formats (using ffmpeg)
    if source.exists():
        print("Converting sample_speech.m4a to other formats...")

        # Convert to WAV
        convert_with_ffmpeg(source, fixtures_dir / 'sample_speech.wav')
        print("  - sample_speech.wav")

        # Convert to FLAC
        convert_to_flac(source, fixtures_dir / 'sample_speech.flac')
        print("  - sample_speech.flac")

        # Convert to MP3
        convert_with_ffmpeg(source, fixtures_dir / 'sample_speech.mp3', bitrate='128k')
        print("  - sample_speech.mp3")

        # Convert to FLAC with different sample rate (44.1kHz -> 16kHz)
        convert_to_flac(fixtures_dir / 'sample_speech.wav', fixtures_dir / 'sample_speech_16k.flac', sample_rate=16000)
        print("  - sample_speech_16k.flac")
    else:
        print(f"  Warning: Source file {source} not found, skipping conversions")

    # 4. Generate edge case files
    print("Generating edge case files...")

    # Empty file
    (fixtures_dir / 'empty.wav').touch()
    print("  - empty.wav (0 bytes)")

    # Invalid file (text with wav extension)
    (fixtures_dir / 'invalid.wav').write_text('This is not a real WAV file')
    print("  - invalid.wav (corrupt file)")

    print("\nDone! Test audio files generated successfully.")


if __name__ == '__main__':
    main()
