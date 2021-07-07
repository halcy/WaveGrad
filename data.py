import numpy as np
np.random.seed(1234)

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import librosa

from utils import parse_filelist


class AudioDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, config, training=True):
        super(AudioDataset, self).__init__()
        self.config = config
        self.hop_length = config.data_config.hop_length
        self.training = training

        if self.training:
            self.segment_length = config.training_config.segment_length
        self.sample_rate = config.data_config.sample_rate

        self.filelist_path = config.training_config.train_filelist_path \
            if self.training else config.training_config.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)

    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)
        # To ensure upsampling/downsampling will be processed in a right way for full signals
        if not self.training:
            p = (audio.shape[-1] // self.hop_length + 1) * self.hop_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        return audio.squeeze(), sample_rate

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio, sample_rate = self.load_audio_to_torch(audio_path)

        if self.training:
            # Normalize and do amplitude augmentation
            audio_max = audio.abs().max()
            audio = audio / audio_max
            vol_aug_factor = float(1.0 - min((torch.randn(1) * 0.5).abs(), 0.8))
            audio = audio * vol_aug_factor

            # Pitch-shift augmentation
            audio = librosa.effects.pitch_shift(audio.numpy(), sample_rate, torch.randn(1) * 2.0)

            # Time-stretch augmentation
            stretch_aug_factor = float(max(min(torch.randn(1) * 0.5 + 1.0, 2.0), 0.5))
            audio = librosa.effects.time_stretch(audio, stretch_aug_factor)

            # Sign flip augmentation (phase inversion)
            audio = audio * (-1.0 if torch.randn(1) > 0.0 else 1.0)

            # Teensy bit of noise aug
            audio += torch.randn(audio.shape).numpy() * 0.0001
            
            # Back to Torch
            audio = torch.Tensor(audio)
            
        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according config."""

        if not self.training:  # If test
            return audio
        # Take segment of audio for training
        if audio.shape[-1] > self.segment_length:
            max_audio_start = audio.shape[-1] - self.segment_length
            audio_start = np.random.randint(0, max_audio_start)
            segment = audio[audio_start:audio_start+self.segment_length]
        else:
            segment = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[-1]), 'constant'
            ).data
        return segment

    def __len__(self):
        return len(self.audio_paths)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""
    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)
    
    def forward(self, x):
        outputs = self.torchaudio_backend(x).log10()
        mask = torch.isinf(outputs)
        outputs[mask] = 0
        return outputs[..., :-1]
