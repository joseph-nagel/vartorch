'''Half moons datamodule.'''

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule


def make_half_moons(num_samples,
                    noise_level=0.15,
                    offsets=(0.15, -0.15),
                    random_state=0,
                    test_size=None):
    '''Create half moons data.'''

    # create data
    x, y = make_moons(
        num_samples,
        shuffle=True,
        noise=abs(noise_level),
        random_state=random_state
    )

    # add offsets
    x[y==0, 1] += offsets[0]
    x[y==1, 1] += offsets[1]

    # return
    if test_size is None:
        return x, y

    # split data and return
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=test_size
        )

        return x_train, x_val, y_train, y_val


class MoonsDataModule(LightningDataModule):
    '''
    DataModule for half moons data.

    Parameters
    ----------
    num_train : int
        Number of training samples.
    num_val : int, optional
        Number of validation samples.
    num_test : int, optional
        Number of testing samples.
    noise_level : float, optional
        Noise standard deviation.
    offsets : tuple, optional
        Offsets applied to the data.
    batch_size : int, optional
        Batch size of the data loader.
    num_workers : int, optional
        Number of workers for the loader.

    '''

    def __init__(self,
                 num_train,
                 num_val=0,
                 num_test=0,
                 noise_level=0.15,
                 offsets=(0.15, -0.15),
                 batch_size=32,
                 num_workers=0):

        super().__init__()

        # set data parameters
        self.num_train = abs(int(num_train))
        self.num_val = abs(int(num_val))
        self.num_test = abs(int(num_test))
        self.noise_level = abs(noise_level)
        self.offsets = offsets

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        '''Prepare numerical data.'''

        # create data
        num_samples = self.num_train + self.num_val + self.num_test

        x, y = make_half_moons(
            num_samples,
            noise_level=self.noise_level,
            offsets=self.offsets,
            random_state=42,
            test_size=None
        )

        # transform to tensor
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    @property
    def x_train(self):
        return self.x[:self.num_train]

    @property
    def y_train(self):
        return self.y[:self.num_train]

    @property
    def x_val(self):
        return self.x[self.num_train:self.num_train+self.num_val]

    @property
    def y_val(self):
        return self.y[self.num_train:self.num_train+self.num_val]

    @property
    def x_test(self):
        return self.x[self.num_train+self.num_val:]

    @property
    def y_test(self):
        return self.y[self.num_train+self.num_val:]

    def setup(self, stage):
        '''Set up train/test/val. datasets.'''

        # create train dataset
        if stage == 'fit':
            self.train_set = TensorDataset(self.x_train, self.y_train)

        # create val. dataset
        elif stage == 'validate':
            self.val_set = TensorDataset(self.x_val, self.y_val)

        # create test dataset
        elif stage == 'test':
            self.test_set = TensorDataset(self.x_test, self.y_test)

    def train_dataloader(self):
        '''Create train dataloader.'''
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self):
        '''Create val. dataloader.'''
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self):
        '''Create test dataloader.'''
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )
