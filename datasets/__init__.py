import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

def create_dataset(config):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(config)
    """
    data_loader = CustomDatasetDataLoader(config)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, config):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.config = config
        dataset_class = find_dataset_using_name(config['dataset_mode'])
        self.dataset = dataset_class(config)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=not config['serial_batches'],
            num_workers=int(config['num_threads']))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.config['max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.config['batch_size >= self.config['max_dataset_size:
                break
            yield data