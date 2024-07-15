import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import butter, filtfilt, resample

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "/content/drive/My Drive/dl_lecture_competition_pub/data", fs=200, new_fs=100, lowcut=1, highcut=50, order=5):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.num_classes = 1854
        
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.fs = fs	
        self.new_fs = new_fs
        
        # データファイルのパスを設定
        self.X_path = os.path.join(data_dir, f"{split}_X.pt")
        self.subject_idxs_path = os.path.join(data_dir, f"{split}_subject_idxs.pt")

        # ラベルファイルの有無を確認し、あればパスを設定
        if split != "test":
            self.y_path = os.path.join(data_dir, f"{split}_y.pt")
            self.y = torch.load(self.y_path)

        # データをロード
        self.X = torch.load(self.X_path)
        self.subject_idxs = torch.load(self.subject_idxs_path)
        
        self.b, self.a = self.butter_bandpass()


    def butter_bandpass(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    # def apply_filter(self, data):
    #     data = np.copy(data)  # 負のストライドを回避するためにコピー
    #     if data.ndim != 2 or data.shape[0] <= 1 or data.shape[1] <= 1:
    #         raise ValueError(f"Invalid data shape for filtering: {data.shape}")
    
    #     return filtfilt(self.b, self.a, data, axis=1)        

    def resample(self, data):
        data = np.copy(data)  # 負のストライドを回避するためにコピー
        num_samples = int(data.shape[1] * self.new_fs / self.fs)
        return resample(data, num_samples, axis=1)

    def scale(self, data):
        return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    def baseline_correction(self, data):
        baseline = np.mean(data[:, :int(self.fs * 0.1)], axis=1, keepdims=True)
        return data - baseline

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].cpu().numpy()
        X_resampled = self.resample(X)
        # X_filtered = self.apply_filter(X_resampled)
        # X_scaled = self.scale(X_filtered)
        X_scaled = self.scale(X_resampled)
        X_corrected = self.baseline_correction(X_scaled)
        X_corrected = np.ascontiguousarray(X_corrected)  # コピーして正のストライドを持つようにする
        X_corrected = torch.tensor(X_corrected, dtype=torch.float32)
 
        if hasattr(self, "y"):
            y = self.y[idx]
            subject_idx = self.subject_idxs[idx]
            return X_corrected, y, subject_idx
        else:
            subject_idx = self.subject_idxs[idx]
            return X_corrected, subject_idx
    
    def __len__(self) -> int:
        return len(self.X)

    # def __getitem__(self, i):
    #     if hasattr(self, "y"):
    #         return self.X[i], self.y[i], self.subject_idxs[i]
    #     else:
    #         return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]