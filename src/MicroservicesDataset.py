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
        for filen in os.listdir():
            print(filen)
        for filename in os.listdir(dataset_path):
            print(filename)
            data_path = os.path.join(dataset_path, filename)
            data = np.loadtxt(data_path, usecols=(4, 5))
            data = data.astype(int)
            uniques = np.unique(data[:, 0], return_counts=True)[1]
            times = np.unique(data[:, 0], return_counts=True)[0]

            if global_min > min(times) or global_min == 0:
                global_min = int(min(times))
            if global_max < max(times) or global_max == 0:
                global_max = int(max(times))

            cumulative = np.cumsum(uniques)
            split_values = np.split(data[:, 1], cumulative[:-1])
            split_values_averaged = [int(sum(vals)/len(vals)) for vals in split_values]
            latencies_by_second = np.column_stack((times, split_values_averaged))
            #latencies_by_second = self._normalize(latencies_by_second)
            data_arr.append(latencies_by_second)
            num_services += 1
            
        #print(num_services)
        #print(data_arr)
        self.service_count = num_services
        data_master = self._construct_master(global_min, global_max, num_services, data_arr)


        labels_arr = []
        for filename in os.listdir(labels_path):
            print(filename)
            label_path = os.path.join(labels_path, filename)
            labels = np.loadtxt(label_path, usecols=(4, 5, 6))
            labels = labels.astype(int)
            uniques = np.unique(labels[:, 0], return_counts=True)[1]
            times = np.unique(labels[:, 0], return_counts=True)[0]

            ### We don't want to recalculate min and max here. Not necessary.

            ### if global_min > min(times) or global_min == 0:
            ###     global_min = min(times)
            ### if global_max < max(times) or global_max == 0:
            ###     global_max = max(times)

            cumulative = np.cumsum(uniques)
            #congested_split_values = np.split(labels[:, 1], cumulative[:-1])
            receiver_split_values = np.split(labels[:, 2], cumulative[:-1])
            #congested_split_values_avg = [sum(vals)/len(vals) for vals in congested_split_values]
            receiver_split_values_avg = [int(sum(vals)/len(vals)) for vals in receiver_split_values]
            receiver_by_second = np.column_stack((times, receiver_split_values_avg))
            #congested_and_receiver_by_second = np.column_stack((receiver_by_second, receiver_split_values_avg))
            #labels_by_second = self._normalize(receiver_by_second)
            labels_arr.append(receiver_by_second)
            
        label_master = self._construct_master(global_min, global_max, num_services, labels_arr)
        
        ### For every latency datapoint at time t, the label is receiver window data at time t+1
        shifted_data = data_master[:-2,:]
        #shifted_labels = label_master[1:-1,:]
        shifted_labels = data_master[1:-1, :]
        
        normalized_data = self._normalize(shifted_data)
        normalized_labels = self._normalize(shifted_labels)
        
        #print(shifted_data.shape)
        #print(shifted_labels.shape)
        # Convert to float32
        self._x = torch.reshape(torch.Tensor(normalized_data), (258, 5, 6))
        self._y = torch.reshape(torch.Tensor(normalized_labels), (258, 5, 6))


    def _construct_master(self, global_min, global_max, num_services, _arr):        
        time_range = self._add_to_time_range(global_min, global_max, num_services, _arr)
        time_range = self._interpolate(time_range)

        print(time_range)
        return time_range


    ### Removes t values, so that time is implicit in array position
    def _add_to_time_range(self, global_min, global_max, num_services, data_arr):
        time_range = np.zeros([global_max - global_min, num_services])
        print(global_min)
        print(global_max)
        for i in range(num_services):
            serv_data = data_arr[i]
            for j in range(global_min, global_max):
                ### serv_data only has data for some timesteps
                if j in serv_data[:, 0]:
                    idx = np.where(serv_data[:, 0] == j)
                    time_range[j - global_min, i] = serv_data[idx, 1]
                else: 
                    time_range[j - global_min, i] = -1
        return time_range

    ### last_non_zero and first_non_zero represent values other than -1, which represent missing latencies
    def _interpolate(self, time_range):
        time_range = np.array(time_range)
        for j in range(time_range.shape[1]):
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
            for i in range(time_range.shape[0]):
                if len(interpolated) == 0:
                    if time_range[i, j] != -1:
                        last_non_zero = time_range[i, j]
                    else:
                        zero_start = i
                        interpolated.append(0)
                else:
                    if time_range[i, j] != -1:
                        first_non_zero = time_range[i, j]
                        ### Because indexing is exclusive, we have zero end be the index of the first nonzero value
                        ### Add 1 to len(interpolated) because that is the number of 'hops' from last_non_zero to first_non_zero
                        zero_end = i
                        incr = (first_non_zero - last_non_zero) / (len(interpolated) + 1)
                        last_val = last_non_zero
                        for x in range(len(interpolated)):
                            interpolated[x] = last_val + incr
                            last_val = interpolated[x]
                        #print("***")
                        #print(interpolated)
                        #print("======")
                        #print(time_range[zero_start:zero_end, j])
                        time_range[zero_start:zero_end, j] = interpolated
                        #print("~~~~~~~~")
                        #print(time_range[zero_start:zero_end, j])
                        last_non_zero = time_range[i, j]
                        interpolated = []
                    else:
                        interpolated.append(0)
                if i == time_range.shape[0] - 1:
                    if len(interpolated) != 0:
                        for x in range(len(interpolated)):
                            interpolated[x] = last_non_zero
                        time_range[zero_start:, j] = interpolated
                    else:
                        if time_range[i, j] == -1:
                            time_range[i, j] = last_non_zero

        return time_range


    def _normalize(self, data):
        ### This can already handle multiple columns
        for i in range(1, data.shape[1]):
            col = data[:, i]
            mean = np.mean(col)
            std = np.std(col)
            data[:, i] = (data[:, i] - mean) / (std + np.finfo(float).eps)
        return data

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
    m = MicroservicesDataset('../data/data/', '../data/labels/')

if __name__ == "__main__":
    main()
