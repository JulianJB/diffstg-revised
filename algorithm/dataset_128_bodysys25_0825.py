# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

def search_recent_data(train_data, window_end_idx, T_p, T_h):
    """
    Validates and creates indices for a single sliding window from the time-series data.
    This function's main purpose is to act as a "window validator." It is called
    repeatedly in a loop to generate a list of all possible valid training samples
    by sliding a window of a fixed size (T_h) across the entire dataset.
    """
    # The original implementation had a check for the label window extending beyond
    # the data array. Since T_p = 0 for imputation, this check isn't strictly necessary
    # but is kept here for completeness
    if window_end_idx + T_p > len(train_data):
        return None

    # Calculate the start index of the history window
    # The window will span from [start_idx, window_end_idx]
    start_idx = window_end_idx - T_h

    # Define the history window's end index (exclusive)
    end_idx = window_end_idx

    # Ensure that the calculated start_idx does not fall before the beginning of the data array (index 0)
    if start_idx < 0:
        return None

    # If the window is valid, return the index pairs
    history_indices = (start_idx, end_idx)

    # For imputation with T_p=0, this will be (window_end_idx, window_end_idx),
    # representing an empty slice. This part of the return value is ignored later
    # in the data loading process
    future_indices = (window_end_idx, window_end_idx + T_p)

    return history_indices, future_indices

class HARDataset(Dataset):
    def __init__(self, data_array, config, dataset_name="HAR_unspecified"):
        self.T_h = config.model.T_h # History length (number of time steps for input)
        self.T_p = 0 # Prediction horizon (number of time steps to predict)
        self.V = config.model.V # Number of vertices (nodes/sensors)

        self.dataset_name = dataset_name

        #  The full data array for this dataset split (e.g., training data)
        self.feature = data_array

        #  Define the bounds for creating sliding windows (the entire length of the data array)
        self.data_range = (0, self.feature.shape[0]) # Start from 0 to the end of the current data_array

        # Prepare samples (sliding windows)
        self.idx_lst = self.create_sliding_window_indices()
        print(f'{self.dataset_name} - sample num: {len(self.idx_lst)}')

    def __getitem__(self, index):
        # Retrieve the start and end indices for the current data window
        history_indices, _ = self.idx_lst[index]

        # Extract the slice of data for the current window -> shape: (T_h, V, F)
        current_window_data = self.feature[history_indices[0]:history_indices[1]]

        # For the imputation task, we need two things from the dataloader:
        # 1. The complete, uncorrupted data window to serve as the ground truth (the "label")
        # 2. The same data window which will be used as the input feature to be masked
        # Therefore, we return the same tensor twice.
        return current_window_data, current_window_data

    def __len__(self):
        return len(self.idx_lst)

    def create_sliding_window_indices(self):
        """
        Creates a master list of all valid sliding window indices for the dataset.

        This function acts as an orchestrator. It iterates through every possible
        end-point for a window within the full time-series data. For each potential
        end-point, it calls `search_recent_data` to validate if a complete window
        can be formed.
        """
        idx_lst = []

        # Define the range of the main iteration loop
        # We will slide the window across this entire data range
        start_iteration_idx = self.data_range[0]
        end_iteration_idx = self.data_range[1]

        # Iterate through every possible end-point for a history window
        for window_end_idx in range(start_iteration_idx, end_iteration_idx):
            # For the current end-point, call the "validator" function to check
            # if a valid window can be created
            recent_sample_indices = search_recent_data(
                train_data=self.feature,  # The data array for this split
                window_end_idx=window_end_idx,
                T_p=self.T_p,  # Prediction horizon (0 for imputation)
                T_h=self.T_h  # History length from config
            )

            # If search_recent_data returns indices (not None), it means the
            # window is valid and should be added to the master list
            if recent_sample_indices:
                idx_lst.append(recent_sample_indices)

        return idx_lst

if __name__ == '__main__':
    pass