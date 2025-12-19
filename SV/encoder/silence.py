import webrtcvad
from scipy.ndimage.morphology import binary_dilation
from params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

import argparse
import os
import sys
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from audioread.exceptions import NoBackendError

from params_model import model_embedding_size as speaker_embedding_size

import sys
sys.path.append('../')
from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.modelutils import check_model_paths
# from vocoder import inference as vocoder




int16_max = (2 ** 15) - 1

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


if __name__ == "__main__":
    num_generated = 0
    wav, source_sr =librosa.load("D:\\Mini Project\\Real-Time-Voice-Cloning-master\\OUTPUT\\15 sec male audio sample\\demo_output_00.wav")
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
    synthesizer = Synthesizer(Path("synthesizer/saved_models/pretrained/pretrained.pt"))
    input_duration = librosa.get_duration(wav,sr=sampling_rate)
    print("input",input_duration)
    generated_wav = trim_long_silences(wav)
    output_duration = librosa.get_duration(generated_wav,sr=sampling_rate)
    print(output_duration)
    print("ratio",output_duration/input_duration)
    filename = "demo_output_%02d.wav" % num_generated
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % filename)

