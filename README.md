# audio-sync

Align a WAV file to each MP4 in a folder and mux the result using waveform cross-correlation.

## Usage

```bash
python3 audio_sync.py --wav audio.wav --videos ./videos/ [--output ./synced/] [--keep-original-audio]
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--wav` | Yes | Path to the master WAV file |
| `--videos` | Yes | Folder containing `.mp4` files |
| `--output` | No | Output folder (default: `<videos>/synced/`) |
| `--keep-original-audio` | No | Mix WAV with original video audio instead of replacing it |

## Setup

### Using a virtual environment (recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install scipy numpy

# Install ffmpeg (macOS)
brew install ffmpeg
```

### Without virtual environment

```bash
pip install scipy numpy
brew install ffmpeg
```

## How It Works

1. Extracts audio from each MP4 video
2. Uses cross-correlation to find where the video's audio appears in the master WAV
3. Calculates sync offset and confidence score
4. Trims/pads the master WAV to match video timing
5. Muxes the synced audio back into each video

## Example

```bash
# Basic usage - replaces original audio
python3 audio_sync.py --wav master.wav --videos ./clips/

# Keep original audio (mixes both)
python3 audio_sync.py --wav master.wav --videos ./clips/ --keep-original-audio

# Custom output folder
python3 audio_sync.py --wav master.wav --videos ./clips/ --output ./output/
```

## Output

For each input video `clips/video.mp4`, creates `synced/video_synced.mp4`.
