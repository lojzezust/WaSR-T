import math
from itertools import islice
from operator import itemgetter
from typing import Optional, Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DistributedSampler, ConcatDataset

class DatasetBatchSampler(torch.utils.data.Sampler):
    """Samples batches by taking a fixed number of samples from each dataset."""

    def __init__(self, concat_dataset: ConcatDataset, num_samples: List[int], shuffle: bool = True, num_batches: Optional[int] = None):
        """
        Args:
            concat_dataset (ConcatDataset): ConcatDataset containing the datasets to sample from
            num_samples (list[int]): number of samples in a batch for each dataset
            shuffle (bool, optional): shuffle each dataset, default: True
            num_batches (int, optional): Number of batches to sample. If not set, the number is computed such that all samples from all datasets are observed.
        """
        self.datasets = []
        start = 0
        num_batches_c = 0
        for ds, n_samples in zip(concat_dataset.datasets, num_samples):
            indices = list(range(start, start+len(ds)))
            self.datasets.append(self._repeat_shuffle(indices, shuffle))
            start += len(ds)

            # Compute number of batches such that all samples are observed
            num_batches_c = max(num_batches_c, math.ceil(len(ds) / n_samples))

        self.num_batches = num_batches_c if num_batches is None else num_batches
        self.num_samples = num_samples

    def _repeat_shuffle(self, indices, shuffle):
        """Infinite iterator over the indices. If shuffle=True, at the beginning of each iteration the indices are shuffled randomly."""
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                yield i

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for n_samples, ds in zip(self.num_samples, self.datasets):
                batch += list(islice(ds, n_samples))

            yield batch

    def __len__(self):
        return self.num_batches
class DatasetRandomSampler(torch.utils.data.Sampler):
    """Takes the next sample from a randomly chosen dataset. Datasets are repeated (and optionally shuffled)."""

    def __init__(self, concat_dataset: ConcatDataset, probabilities: List[float], num_samples: int, shuffle: bool = True):
        """
        Args:
            concat_dataset (ConcatDataset): ConcatDataset containing the datasets to sample from
            probabilities (list[float]): probability for sampling each of the datasets
            num_samples (int): the total number of samples to draw from the datasets
            shuffle (bool, optional): shuffle each dataset, default: True
        """
        self.datasets = []
        start = 0
        for ds in concat_dataset.datasets:
            indices = list(range(start, start+len(ds)))
            self.datasets.append(self._repeat_shuffle(indices, shuffle))
            start += len(ds)

        self.probabilities = probabilities
        self.num_samples = num_samples

    def _repeat_shuffle(self, indices, shuffle):
        """Infinite iterator over the indices. If shuffle=True, at the beginning of each iteration the indices are shuffled randomly."""
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                yield i

    def __iter__(self):
        for _ in range(self.num_samples):
            ds_i = np.random.choice(len(self.datasets), p=self.probabilities)
            yield next(self.datasets[ds_i])

    def __len__(self):
        return self.num_samples

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
