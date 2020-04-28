import json
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import torch
import os


class OzeDataset(Dataset):
    """Torch dataset for Oze datachallenge.
    Load dataset from a single npz file.
    Attributes
    ----------
    labels: :py:class:`dict`
        Ordered labels list for R, Z and X.
    Parameters
    ---------
    dataset_path:
        Path to the dataset inputs as npz.
    labels_path:
        Path to the labels, divided in R, Z and X, in json format.
        Default is "labels.json".
    normalize:
        Data normalization method, one of ``'mean'``, ``'max'``.
        Default is ``'max'``.
    """

    def __init__(self,
                 dataset_path: str,
                 labels_path: Optional[str] = "labels.json",
                 normalize: Optional[str] = "max",
                 **kwargs):
        """Load dataset from npz."""
        super().__init__(**kwargs)

        self._normalize = normalize

        self._load_txt(dataset_path, labels_path)

    def _load_txt(self, dataset_path, labels_path):
        # Load dataset as csv
        dataset = np.load(dataset_path)
        data_dict = dict()
        for filename in os.listdir(dataset_path):
            data_path = os.path.join(dataset_path, filename)
            data_dict[filename] = np.loadtxt(data_path, ncols=(4, 5))
        
        labels_dict = dict()
        for filename in os.listdir(labels_path):
            labels_path = os.path.join(labels_path, filename)
            labels_dict[filename] = np.loadtxt(labels_path, ncols=(3, 4, 5))
        
        
        # Convert to float32
        self._x = torch.Tensor(self._x)
        self._y = torch.Tensor(self._y)

    def rescale(self,
                y: np.ndarray,
                idx_label: int) -> torch.Tensor:
        """Rescale output from initial normalization.
        Arguments
        ---------
        y:
            Array to resize, of shape (K,).
        idx_label:
            Index of the output label.
        """
        if self._normalize == "max":
            return y * (self._M[idx_label] - self._m[idx_label] + np.finfo(float).eps) + self._m[idx_label]
        elif self._normalize == "mean":
            return y * (self._std[idx_label] + np.finfo(float).eps) + self._mean[idx_label]
        else:
            raise(
                NameError(f'Normalize method "{self._normalize}" not understood.'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]


class OzeDatasetWindow(OzeDataset):
    """Torch dataset with windowed time dimension.
    Load dataset from a single npz file.
    Attributes
    ----------
    labels: :py:class:`dict`
        Ordered labels list for R, Z and X.
    Parameters
    ---------
    dataset_x:
        Path to the dataset inputs as npz.
    labels_path:
        Path to the labels, divided in R, Z and X, in json format.
        Default is "labels.json".
    window_size:
        Size of the window to apply on time dimension.
        Default 5.
    padding:
        Padding size to apply on time dimension windowing.
        Default 1.
    """

    def __init__(self,
                 dataset_path: str,
                 labels_path: Optional[str] = "labels.json",
                 window_size: Optional[int] = 5,
                 padding: Optional[int] = 1,
                 **kwargs):
        """Load dataset from npz."""
        super().__init__(dataset_path, labels_path, **kwargs)

        self._window_dataset(window_size=window_size, padding=padding)

    def _window_dataset(self, window_size=5, padding=1):
        m, K, d_input = self._x.shape
        _, _, d_output = self._y.shape

        step = window_size - 2 * padding
        n_step = (K - window_size - 1) // step + 1

        dataset_x = np.empty(
            (m, n_step, window_size, d_input), dtype=np.float32)
        dataset_y = np.empty((m, n_step, step, d_output), dtype=np.float32)

        for idx_step, idx in enumerate(range(0, K-window_size, step)):
            dataset_x[:, idx_step, :, :] = self._x[:, idx:idx+window_size, :]
            dataset_y[:, idx_step, :, :] = self._y[:,
                                                   idx+padding:idx+window_size-padding, :]

        self._x = dataset_x
        self._y = dataset_y