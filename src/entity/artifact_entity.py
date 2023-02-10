from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataIngestionArtifact:
    train_dataset: Dataset

    test_dataset: Dataset


@dataclass
class DataLoaderArtifact:
    train_dataloader: DataLoader

    test_dataloader: DataLoader
