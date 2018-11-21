import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import datasets, transforms


class TransformedDataset(data.Dataset):
    """ Wraps a given dataset and applies the specified transformations to the samples """

    def __init__(self, dataset, transform):
        """
        Args:
            dataset (Dataset): Image dataset that provides the samples.
            transform (callable): Transform to be applied on a sample.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


def train_test_split_dataset(full_dataset, test_size=0.2):
    """ Splits the given dataset in a train and test set using stratified sampling """
    full_indices = np.arange(len(full_dataset))
    full_targets = np.array([target for _, target in full_dataset.samples])

    train_indices, test_indices = train_test_split(full_indices, test_size=test_size, random_state=42,
                                                   stratify=full_targets)
    return data.Subset(full_dataset, train_indices), data.Subset(full_dataset, test_indices)


