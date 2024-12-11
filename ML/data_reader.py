import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CSVDataSet(Dataset):
    """
    A custom Dataset class to read data from a CSV file.
    """

    def __init__(self, filepath):
        """
        Initialize the dataset by loading the data from a CSV file.

        Args:
        filepath (str): The path to the CSV file.
        """
        self.data = pd.read_csv(filepath)
     #   print("Original data shape:", self.data.shape)  # 打印原始数据形状
        self.data = self.data.values

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Generate one sample of data.

        Args:
        index (int): The index of the sample to return.

        Returns:
        torch.Tensor: The sample as a PyTorch tensor.
        """
        # Return the data sample at the specified index as a float tensor
        item = torch.tensor(self.data[index], dtype=torch.float32)
       # print("Item shape on load:", item.shape)  # 打印加载单项数据时的形状
        return item


def get_data_loader(filepath, batch_size=1024):
    """
    Create a DataLoader to batch and shuffle the data from a CSV file.

    Args:
    filepath (str): The path to the data file.
    batch_size (int): The number of samples in each batch.

    Returns:
    DataLoader: A DataLoader object that provides batches of data.
    """
    dataset = CSVDataSet(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
