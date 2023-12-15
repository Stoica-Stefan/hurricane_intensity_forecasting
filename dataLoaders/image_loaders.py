from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

# Statistics about input data and labels
DATA_MEAN = 271.14
DATA_STD = 25.28

LABELS_MEAN = 41.31
LABELS_STD = 23.12

class ImageDataset(Dataset):
    """Dataset of Hurricane Images from all the given years."""

    def __init__(self, dataset_dir, years, transform=None, standardization = False):
        """
        Args:
            dataset_dir (string): Directory of the entire dataset.
            year (int): the year from which to get the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset_dir = dataset_dir
        self.years = years
        self.transform = transform
        self.standardization = standardization

        self.sample_paths = []
        self.nr_images = 0

        for y in self.years:
            with open(os.path.join(self.dataset_dir, str(y), 'index.data'), 'rb') as f:
                year_info = pickle.load(f)

                for h in year_info['hurricanes']:
                    self.sample_paths += h['image_paths']
                    self.nr_images += h['nr_images']

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx):
        sample_path = os.path.join(self.dataset_dir, self.sample_paths[idx])

        data, label = 0, 0
        with open(sample_path, 'rb') as sample_file:
            sample = pickle.load(sample_file)
            data = sample['data']
            label = sample['label']

        data = torch.Tensor(np.array([data]))
        label = torch.Tensor(np.array([label]))

        if self.transform:
            data = self.transform(data)

        if self.standardization:
            data = (data - DATA_MEAN) / DATA_STD
            label = (label - LABELS_MEAN) / LABELS_STD

        return data, label

class ImageDataset_Testing(Dataset):
    """Testing dataset of Hurricane Images from all the given years.
        Differs from training dataset as it returns arrays containing entire hurricanes.
    """

    def __init__(self, dataset_dir, years, transform=None, standardization = False):
        """
        Args:
            dataset_dir (string): Directory of the entire dataset.
            years (int): the years from which to get the images
            transform (callable, optional): Optional transform to be applied
                on data samples.
        """

        self.dataset_dir = dataset_dir
        self.years = years
        self.transform = transform
        self.standardization = standardization

        self.hurricane_sample_paths = []
        self.nr_hurricanes = 0
        self.nr_images = 0

        for y in self.years:
            with open(os.path.join(self.dataset_dir, str(y), 'index.data'), 'rb') as f:

                year_info = pickle.load(f)

                for h in year_info['hurricanes']:
                    if h['nr_images'] > 0:
                        self.nr_hurricanes += 1
                        self.hurricane_sample_paths.append([])

                        last_time = np.datetime64("1970-01-01")

                        for sample_path in h['image_paths']:
                            with open(os.path.join(self.dataset_dir, sample_path), 'rb') as sample_file:
                                sample = pickle.load(sample_file)
                                if sample['time'] > last_time:
                                    last_time = sample['time']

                                    self.hurricane_sample_paths[-1].append(sample_path)
                                    self.nr_images += 1

    def get_nr_images(self):
        return self.nr_images

    def __len__(self):
        return self.nr_hurricanes

    def __getitem__(self, idx):

        sample_paths = [os.path.join(self.dataset_dir, path) for path in self.hurricane_sample_paths[idx]]

        data, labels = [], []
        for path in sample_paths:
            with open(path, 'rb') as sample_file:
                sample = pickle.load(sample_file)
                data.append([sample['data']])
                labels.append([sample['label']])

        data = torch.Tensor(np.array(data))
        labels = torch.Tensor(np.array(labels))

        if self.transform:
            data = self.transform(data)

        if self.standardization:
            data = (data - DATA_MEAN) / DATA_STD
            labels = (labels - LABELS_MEAN) / LABELS_STD

        return data, labels