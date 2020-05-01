import json
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import torch
import os


class MicroservicesDataset(Dataset):
    """
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
                 labels_path: str,
                 **kwargs):
        super().__init__(**kwargs)

        self._load_txt(dataset_path, labels_path)

    def _load_txt(self, dataset_path, labels_path):
        # Load dataset as csv
        data_arr = []
        global_min = 0
        global_max = 0
        num_services = 0
        for filename in os.listdir(dataset_path):
            data_path = os.path.join(dataset_path, filename)
            data = np.loadtxt(data_path, usecols=(4, 5))
            uniques = np.unique(data[:, 0], return_counts=True)[1]
            times = np.unique(data[:, 0], return_counts=True)[0]

            if global_min > min(times) or global_min == 0:
                global_min = min(times)
            if global_max < max(times) or global_max == 0:
                global_max = max(times)

            cumulative = np.cumsum(uniques)
            split_values = np.split(data[:, 1], cumulative[:-1])
            split_values_averaged = [sum(vals)/len(vals) for vals in split_values]
            latencies_by_second = np.column_stack((times, split_values_averaged))
            latencies_by_second = self._normalize(latencies_by_second)
            data_arr.append(latencies_by_second)
            num_services += 1
            
        self.service_count = num_services
        data_master = self._construct_master(global_min, global_max, num_services, data_arr)


        labels_arr = []
        for filename in os.listdir(labels_path):
            labels_path = os.path.join(labels_path, filename)
            labels = np.loadtxt(data_path, usecols=(4, 5, 6))
            uniques = np.unique(labels[:, 0], return_counts=True)[1]
            times = np.unique(labels[:, 0], return_counts=True)[0]

            ### We don't want to recalculate min and max here. Not necessary.

            ### if global_min > min(times) or global_min == 0:
            ###     global_min = min(times)
            ### if global_max < max(times) or global_max == 0:
            ###     global_max = max(times)

            cumulative = np.cumsum(uniques)
            congested_split_values = np.split(labels[:, 1], cumulative[:-1])
            receiver_split_values = np.split(labels[:, 2], cumulative[:-1])
            congested_split_values_avg = [sum(vals)/len(vals) for vals in congested_split_values]
            receiver_split_values_avg = [sum(vals)/len(vals) for vals in receiver_split_values]
            receiver_by_second = np.column_stack((times, congested_split_values_avg))
            congested_and_receiver_by_second = p.column_stack((receiver_by_second, receiver_split_values_avg))
            labels_by_second_normalized = self._normalize(congested_and_receiver_by_second)
            labels_arr.append(labels_by_second_normalized)
            
        label_master = self._construct_master(global_min, global_max, num_services, data_arr, labels=True)

        # Convert to float32
        self._x = torch.Tensor(data_master)
        self._y = torch.Tensor(label_master)


    def _construct_master(self, global_min, global_max, num_services, data_arr, labels=False):        
        time_range = self._add_to_time_range(global_min, global_max, num_services, data_arr, labels)
        if labels:
            time_range = self._interpolate(time_range)
        else:
            time_range = self._interpolate_latencies(time_range)

        return time_range


    ### Removes t values, so that time is implicit in array position
    def _add_to_time_range(self, global_min, global_max, num_services, data_arr, labels):
        time_range = [[] for i in range(global_min, global_max)]
        for i in range(num_services):
            serv_data = np.array(data_arr[i])
            for j in range(global_min, global_max):
                ### serv_data only has data for some timesteps
                if j in serv_data[:, 0]:
                    idx = serv_data[:, 0].index(j)
                    if labels:
                        time_range[global_max - j].append([serv_data[idx, 1], serv_data[idx, 2]])
                    else:
                        time_range[global_max - j].append(serv_data[idx, 1])
                else: 
                    if labels:
                        time_range[global_max - j].append([-1, -1])
                    else:
                        time_range[global_max - j].append(-1)
        return time_range

    def _interpolate_latencies(self, time_range):
        time_range = np.array(time_range)
        for j in time_range.shape[1]:
            ### Value
            last_non_zero = 0
            ### Index
            zero_start = 0
            ### Index
            zero_end = 0
            ### Values
            interpolated = []
            ### Value
            first_non_zero = 0
            for i in time_range.shape[0]:
                if len(interpolated) == 0:
                    if time_range[i, j] != -1:
                        last_non_zero = time_range[i, j]
                    else:
                        zero_start = i
                        interpolated.append(0)
                else:
                    if time_range[i, j] != 0:
                        first_non_zero = time_range[i, j]
                        ### Because indexing is exclusive, we have zero end be the index of the first nonzero value
                        ### Add 1 to len(interpolated) because that is the number of 'hops' from last_non_zero to first_non_zero
                        zero_end = i
                        incr = (first_non_zero - last_non_zero) / (len(interpolated) + 1)
                        last_val = last_non_zero
                        for x in range(len(interpolated)):
                            interpolated[x] = last_val + incr
                            last_val = interpolated[x]
                        time_range[zero_start:zero_end, j] = interpolated
                        interpolated = []
                    else:
                        interpolated.append(0)
        return time_range

    
    def _interpolate_windows(self, time_range):
        time_range = np.array(time_range)
        for j in time_range.shape[1]:
            ### Value
            last_non_zero = [0, 0]
            ### Index
            zero_start = 0
            ### Index
            zero_end = 0
            ### Values
            interpolated = []
            ### Value
            first_non_zero = [0, 0]
            for i in time_range.shape[0]:
                if len(interpolated) == 0:
                    if time_range[i, j] != [-1, -1]:
                        last_non_zero = time_range[i, j]
                    else:
                        zero_start = i
                        interpolated.append([0, 0])
                else:
                    if time_range[i, j] != [-1, -1]:
                        first_non_zero = time_range[i, j]
                        ### Because indexing is exclusive, we have zero end be the index of the first nonzero value
                        ### Add 1 to len(interpolated) because that is the number of 'hops' from last_non_zero to first_non_zero
                        zero_end = i
                        incr = (first_non_zero - last_non_zero) / (len(interpolated) + 1)
                        last_val = last_non_zero
                        for x in range(len(interpolated)):
                            interpolated[x] = last_val + incr
                            last_val = interpolated[x]
                        time_range[zero_start:zero_end, j] = interpolated
                        interpolated = []
                    else:
                        interpolated.append([0, 0])
        return time_range


    def _normalize(self, data):
        times = [int(x[0]) for x in data]
        ### This can already handle multiple columns
        for i in range(1, data.shape[1]):
            col = data[:, i]
            mean = np.mean(col)
            std = np.std(col)
            values = (col - mean) / (std + np.finfo(float).eps)
            times = np.column_stack((times, values))
        return times

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

    def get_service_count(self):
        return self.service_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]


def main():
    m = MicroservicesDataset('data/data/', 'data/labels/')

if __name__ == "__main__":
    main()