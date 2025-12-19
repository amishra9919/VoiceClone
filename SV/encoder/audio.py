from scipy.ndimage.morphology import binary_dilation
from SV.encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1
sampling_rate = 16000
audio_norm_target_dBFS = -30 #-30 is for quiter signal 


def  preprocess_wav(fpath_or_wav ,
                   source_sr = None,
                   normalize = True,
                   trim_silence = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else: 
        #if done using browse audio else triggers because the fpath_or_wav for first occurence is neither str nor path 
        #basically it loads if already not loaded into waveforms and sampling

        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        """
        will resample only for the first time
        this resamples the incoming sampling rate (source_sr=22050 hz) to sampling_rate(16000hz)
        here duration = len(wav)/sampling rate. -{ sr wave per second in total len(wav) }
        so here in wav resampling ... the sampling are changed that means len(wav) will also be changed as need to maintain the duration (as there in divison)
        """
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate) 
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

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
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
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


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    # 10 * np.log10(np.mean(wav ** 2)) is current_dBFS calculated in Power (log to base 10)
    """
    Case 1 : To boost quiet signals, never reduce loud ones.
        a) Signal is too loud: 
            dBFS_change < 0  
            increase_only = True
            Condition becomes: (True and True) → True ➡ Return wav unchanged
            ✔ Prevents decreasing volume
            ✔ Loud signals stay loud
        b) Signal is too quiet:
            dBFS_change > 0
            increase_only = True
            Condition becomes: (False and True) → False ➡ ➡ Normalization continues → signal is boosted
    Case 2 : To boost quiet signals, never reduce loud ones.
        a) Signal is too quiet: 
            dBFS_change > 0  
            decrease_only = True
            Condition becomes: (True and True) → True ➡ Return wav unchanged
            ✔ Prevents boosting noise
            ✔ Quiet signals stay quiet
        b) Signal is too loud:
            dBFS_change < 0
            decrease_only = True
            Condition becomes: (False and True) → False ➡ ➡ Normalization continues → signal is attenuated(smaller)

    below code is written for Case 1
    """
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))
