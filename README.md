# VoiceClone

A Python-based voice cloning application that allows users to clone voices from audio samples and generate speech in multiple languages with translation capabilities.

## Features

- **Voice Cloning**: Clone voices from audio files using deep learning models
- **Multi-language Support**: Supports English, Afrikaans, Hmong, Indonesian, Japanese, and Somali
- **Real-time Translation**: Translate text between supported languages
- **Audio Recording**: Record audio directly through the application
- **Noise Reduction**: Automatic noise reduction for better audio quality
- **GUI Interface**: User-friendly Tkinter-based interface

## Project Structure

```
VoiceClone/
├── clone_ui.py              # Main GUI application
├── requirements.txt         # Python dependencies
├── SV/                      # Speaker Verification module
│   ├── encoder/            # Voice encoder models and utilities
│   └── samples/            # Sample audio files
├── TTS/                    # Text-to-Speech module
│   ├── synthesizer/        # Speech synthesis models
│   └── vocoder/           # Audio vocoder models
├── Audio samples/          # Input audio samples
├── input/                  # Input audio directory
├── Input Sample/           # Sample input files
└── Output/                 # Generated output files
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd VoiceClone
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required model files:
   - `SV/encoder/new_saved_models/encoder.pt`
   - `TTS/synthesizer/saved_models/pretrained/pretrained.pt`
   - `TTS/vocoder/saved_models/pretrained/pretrained.pt`

## Usage

1. Run the application:
```bash
python clone_ui.py
```

2. Follow the GUI steps:
   - **Step 1**: Browse and select an audio file for voice cloning
   - **Step 2**: Enter text and select input language
   - **Step 3**: Choose target language and translate text
   - **Step 4**: Generate cloned voice output

## Dependencies

Key dependencies include:
- `librosa` - Audio processing
- `torch` - Deep learning framework
- `soundfile` - Audio file I/O
- `tkinter` - GUI framework
- `noisereduce` - Noise reduction
- `translate` - Translation services
- `sounddevice` - Audio recording

## Models

The application uses three main models:
- **Encoder**: Converts audio to voice embeddings
- **Synthesizer**: Generates mel spectrograms from text and embeddings
- **Vocoder**: Converts spectrograms to audio waveforms

## Output

Generated audio files are saved in the `output/` directory as `cloned_output00.wav`.

## Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A

## Languages Supported

- English
- Afrikaans
- Hmong
- Indonesian
- Japanese
- Somali

## License

See individual module licenses in `TTS/synthesizer/LICENSE.txt` and `TTS/vocoder/LICENSE.txt`.