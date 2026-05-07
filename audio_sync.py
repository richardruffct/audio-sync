#!/usr/bin/env python3
"""
sync_audio.py — Align a WAV file to each MP4 in a folder and mux the result.

Usage:
    python3 sync_audio.py --wav audio.wav --videos ./videos/ [--output ./synced/] [--keep-original-audio]

Requirements:
    pip install scipy numpy
    brew install ffmpeg
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil

def check_dependencies():
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install it with: brew install ffmpeg")
        sys.exit(1)
    try:
        import scipy
        import numpy
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install scipy numpy")
        sys.exit(1)

def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())

def extract_audio_as_wav(input_path, out_wav_path, sample_rate=16000):
    """Extract/convert audio to a mono WAV at a fixed sample rate."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            out_wav_path
        ],
        capture_output=True
    )

def load_wav_as_array(wav_path):
    import numpy as np
    from scipy.io import wavfile
    rate, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    return rate, data

def find_offset(master_wav, clip_wav):
    """
    Find where clip_wav appears inside master_wav using cross-correlation.
    Returns (offset_seconds, confidence).
    Positive offset = clip starts this many seconds into the master WAV.
    """
    import numpy as np
    from scipy.signal import correlate, resample

    rate_m, master = load_wav_as_array(master_wav)
    rate_c, clip = load_wav_as_array(clip_wav)

    # Resample clip to master sample rate if needed
    if rate_c != rate_m:
        new_len = int(len(clip) * rate_m / rate_c)
        clip = resample(clip, new_len)

    # Normalise both signals
    master = master / (np.std(master) + 1e-9)
    clip = clip / (np.std(clip) + 1e-9)

    # Cross-correlate clip against master
    correlation = correlate(master, clip, mode='full')
    lag = np.argmax(correlation) - (len(clip) - 1)

    offset_seconds = lag / rate_m
    confidence = float(np.max(correlation)) / len(clip)

    return offset_seconds, confidence

def trim_and_mux(wav_path, offset_seconds, duration_seconds, video_path, output_path, keep_original_audio):
    tmp_files = []
    try:
        if offset_seconds >= 0:
            start = offset_seconds
            pad_seconds = 0.0
        else:
            start = 0.0
            pad_seconds = abs(offset_seconds)

        trimmed_wav = tempfile.mktemp(suffix="_trimmed.wav")
        tmp_files.append(trimmed_wav)

        if pad_seconds > 0:
            silence_file = tempfile.mktemp(suffix="_silence.wav")
            tmp_files.append(silence_file)
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=mono",
                "-t", str(pad_seconds), silence_file
            ], capture_output=True)

            partial_wav = tempfile.mktemp(suffix="_partial.wav")
            tmp_files.append(partial_wav)
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", "0",
                "-t", str(max(0, duration_seconds - pad_seconds)),
                partial_wav
            ], capture_output=True)

            list_file = tempfile.mktemp(suffix="_list.txt")
            tmp_files.append(list_file)
            with open(list_file, "w") as f:
                f.write(f"file '{silence_file}'\nfile '{partial_wav}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, trimmed_wav
            ], capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(start),
                "-t", str(duration_seconds),
                "-acodec", "pcm_s16le",
                trimmed_wav
            ], capture_output=True)

        if keep_original_audio:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", trimmed_wav,
                "-filter_complex", "amix=inputs=2:duration=first",
                "-c:v", "copy", "-shortest",
                output_path
            ], capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", trimmed_wav,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ], capture_output=True)

    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)

def process_video(wav_path, video_path, output_dir, keep_original_audio):
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_synced{ext}")

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    tmp_master_wav = tempfile.mktemp(suffix="_master.wav")
    tmp_clip_wav = tempfile.mktemp(suffix="_clip.wav")

    try:
        print("  [1/3] Extracting audio from video...")
        extract_audio_as_wav(video_path, tmp_clip_wav)
        extract_audio_as_wav(wav_path, tmp_master_wav)

        print("  [2/3] Finding sync offset via cross-correlation...")
        offset, confidence = find_offset(tmp_master_wav, tmp_clip_wav)

        duration = get_video_duration(video_path)
        print(f"  ✓ Offset: {offset:+.3f}s | Confidence: {confidence:.2f} | Duration: {duration:.3f}s")

        if confidence < 0.01:
            print(f"  ⚠ Low confidence — result may not be accurate.")

        print("  [3/3] Trimming WAV and muxing into new video...")
        trim_and_mux(wav_path, offset, duration, video_path, output_path, keep_original_audio)

        print(f"  ✓ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {filename}: {e}")
        import traceback; traceback.print_exc()
        return False

    finally:
        for f in [tmp_master_wav, tmp_clip_wav]:
            if os.path.exists(f):
                os.unlink(f)

def main():
    parser = argparse.ArgumentParser(
        description="Sync a WAV file to each MP4 in a folder using waveform cross-correlation."
    )
    parser.add_argument("--wav", required=True, help="Path to the master WAV file")
    parser.add_argument("--videos", required=True, help="Folder containing .mp4 files")
    parser.add_argument("--output", default=None, help="Output folder (default: <videos>/synced/)")
    parser.add_argument("--keep-original-audio", action="store_true",
                        help="Mix WAV with original video audio instead of replacing it")
    args = parser.parse_args()

    check_dependencies()

    wav_path = os.path.abspath(args.wav)
    videos_dir = os.path.abspath(args.videos)
    output_dir = os.path.abspath(args.output) if args.output else os.path.join(videos_dir, "synced")

    if not os.path.isfile(wav_path):
        print(f"ERROR: WAV file not found: {wav_path}")
        sys.exit(1)
    if not os.path.isdir(videos_dir):
        print(f"ERROR: Videos folder not found: {videos_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted([
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.lower().endswith(".mp4")
    ])

    if not mp4_files:
        print(f"No .mp4 files found in: {videos_dir}")
        sys.exit(1)

    print(f"\nWAV file : {wav_path}")
    print(f"Videos   : {videos_dir} ({len(mp4_files)} file(s))")
    print(f"Output   : {output_dir}")

    success, failed = 0, 0
    for video_path in mp4_files:
        ok = process_video(wav_path, video_path, output_dir, keep_original_audio=args.keep_original_audio)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done! {success} succeeded, {failed} failed.")
    print(f"Output folder: {output_dir}")

if __name__ == "__main__":
    main()