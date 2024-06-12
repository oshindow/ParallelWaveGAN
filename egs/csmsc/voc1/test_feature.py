import librosa
import torch
import numpy as np

def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    print("parallel wavegan, librosa.stft", x_stft.shape, x_stft.max(), x_stft.min())
    print(x_stft)
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    print("parallel wavegan librosa.filters.mel:", mel_basis.shape, mel_basis.max(), mel_basis.min())
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))
    print("parallel wavegan, before log", mel.shape, mel.max(), mel.min())
    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0: ###  output log10
        return np.log10(mel) 
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")

sampling_rate = 16000
hop_size = 256
win_length = 1024
fft_size = 1024
num_mels = 80
fmin = 80
fmax = 7600
window = "hann"
filepath = '/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/009901.16k.wav'
audio, sr = librosa.load(filepath, sr=sampling_rate)

####### parallel wavegan
mel = logmelfilterbank(
    audio,
    sampling_rate=sampling_rate,
    hop_size=hop_size,
    fft_size=fft_size,
    win_length=win_length,
    window=window,
    num_mels=num_mels,
    fmin=fmin,
    fmax=fmax,
)
print(audio.shape, audio.max(), audio.min())
print('parallel audio:', audio)

audio = np.pad(audio, (0, fft_size), mode="edge")
audio = audio[: len(mel) * hop_size]
assert len(mel) * hop_size == len(audio)


print(mel.shape, mel.max(), mel.min())
print("parallel mel:", mel)
####### grad-tts

from meldataset import mel_spectrogram
def get_mel(filepath): 
    # audio, sr = ta.load(filepath)
    audio, sr = librosa.load(filepath, sr=sampling_rate)
    # print(audio, audio_l, sr_l)
    audio = torch.from_numpy(audio).unsqueeze(0)
    assert sr == sampling_rate

    # 1. cancel padding inside mel_spectrogram, 2. set center=True, and 3. np.log10
    # then gradtts same as parallel wavegan
    mel = mel_spectrogram(audio, fft_size, num_mels, sampling_rate, hop_size,
                            win_length, fmin, fmax, center=True).squeeze()
    return mel, audio 

mel, audio = get_mel(filepath)
print(audio.shape, torch.max(audio), torch.min(audio))
print('gradtts audio:', audio)

# mel = np.log(mel.numpy())
# print(mel.shape, mel.max(), mel.min())
print(mel.shape, torch.max(mel), torch.min(mel))
print("gradtts mel:", mel)