# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F
from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


class DFT(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, n_fft=1024, hop_size=256, win_size=None, device=None):
        """Initilize ISTFT module.

        
        Args:
            n_fft (int): The number of subbands.
            hop_size (int): The number of filter taps.
            win_size (int): Cut-off frequency ratio.
            device (device): Beta coefficient for kaiser window.

        """
        super(DFT, self).__init__()

        self.n_fft = n_fft
        self.hop_size = hop_size
        if win_size is None:
            self.win_size = n_fft
        else:
            self.win_size = win_size
        self.window = torch.hann_window(self.win_size).to(device)

    def stft(self, x):
        """Perform STFT and convert to magnitude spectrogram.

        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.

        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

        """
        if is_pytorch_17plus:
            x_stft = torch.stft(
                x, self.n_fft, self.hop_size, self.win_size, self.window, return_complex=False
            )
        else:
            x_stft = torch.stft(x, self.n_fft, self.hop_size, self.win_size, self.window)
        # real = x_stft[..., 0]
        # imag = x_stft[..., 1]

        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        return x_stft
        # return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


    def istft(self, stft_spec):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        y_g = torch.istft(stft_spec, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.window, center=True)
        return y_g

# wavefile = '/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k.subband/predictions/19000steps/1_ref.wav'
wavefile = '/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_16k_csmsc_disc_100k/wav/checkpoint-400000steps/eval_16k/csmsc_009901_gen.wav'
wavefile = '/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/exp/train_nodev_16k_csmsc_parallel_wavegan.v1.16k/wav/checkpoint-400000steps/eval_16k/csmsc_009901_gen.wav'
wavefile = '/data2/xintong/parallel_wavegan_downloads/CSMSC/Wave/009901.wav'
import librosa
import torch
y, sr = librosa.load(wavefile, sr=16000) # y: max:0.52, min: -0.4, (34928,)
wave = torch.from_numpy(y).unsqueeze(0).unsqueeze(0) # torch.Size([1, 34928])

print(wave.shape)
x_stft = DFT(device=None).stft(wave.squeeze(1))
print("x_stft:", x_stft.shape)

# x_stft = torch.rand(6, 513, 100, 2)
x_stft_low = torch.nn.functional.pad(x_stft[:,:129,:], pad=(0,0,0,0,0,513-129))
print(x_stft_low, x_stft_low.shape)
x_low = DFT(device=None).istft(x_stft_low)

x_stft_mid = torch.nn.functional.pad(x_stft[:,129:384,:], pad=(0,0,0,0,129,513-384))
print(x_stft_mid, x_stft_mid.shape)
x_mid = DFT(device=None).istft(x_stft_mid)

x_stft_high = torch.nn.functional.pad(x_stft[:,384:,:], pad=(0,0,0,0,384,0))
print(x_stft_high, x_stft_high.shape)
x_high = DFT(device=None).istft(x_stft_high)

import soundfile as sf
print(x_low.shape)
# sf.write('y_.wav', x_low, 24000)
sf.write('y0_ref.wav', x_low[0], 16000)
sf.write('y1_ref.wav', x_mid[0], 16000)
sf.write('y2_ref.wav', x_high[0], 16000)
# sf.write('y3.wav', subband[0,3], 16000)