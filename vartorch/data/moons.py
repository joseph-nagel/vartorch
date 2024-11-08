'''Half-moons datamodule.'''

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule


def make_half_moons(
    num_samples: int,
    noise_level: float = 0.15,
    offsets: tuple[float, float] = (0.15, -0.15),
    random_state: int | None = None,
    test_size: int | float | None = None
):
    '''
    Create half-moons data.

    Parameters
    ----------
    num_samples : int
        Number of samples to create.
    noise_level : float
        Noise standard deviation.
    offsets : (float, float)
        Class-specific offsets.
    random_state : int or None
        Random generator seed.
    test_size : int, float or None
        Test size parameter.

    '''

    # create data
    x, y = make_moons(
        num_samples,
        shuffle=True,
        noise=abs(noise_level),
        random_state=random_state
    )

    # center data
    x[:,0] -= 0.5
    x[:,1] -= 0.25

    # add class-specific offsets
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
    DataModule for half-moons data.

    Parameters
    ----------
    num_train : int
        Number of training samples.
    num_val : int
        Number of validation samples.
    num_test : int
        Number of testing samples.
    noise_level : float
        Noise standard deviation.
    offsets : (float, float)
        Offsets applied to the data.
    random_state : int or None
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        num_train: int,
        num_val: int = 0,
        num_test: int = 0,
        noise_level: float = 0.15,
        offsets: tuple[float, float] = (0.15, -0.15),
        random_state: int | None = 42,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        super().__init__()

        # set data parameters
        self.num_train = abs(int(num_train))
        self.num_val = abs(int(num_val))
        self.num_test = abs(int(num_test))
        self.noise_level = abs(noise_level)
        self.offsets = offsets

        # set random state
        self.random_state = random_state

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        '''Prepare numerical data.'''

        # create data
        num_samples = self.num_train + self.num_val + self.num_test

        x, y = make_half_moons(
            num_samples,
            noise_level=self.noise_level,
            offsets=self.offsets,
            random_state=self.random_state,
            test_size=None
        )

        # transform to tensor
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    @property
    def x_train(self) -> torch.Tensor:
        return self.x[:self.num_train]

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[:self.num_train]

    @property
    def x_val(self) -> torch.Tensor:
        return self.x[self.num_train:self.num_train+self.num_val]

    @property
    def y_val(self) -> torch.Tensor:
        return self.y[self.num_train:self.num_train+self.num_val]

    @property
    def x_test(self) -> torch.Tensor:
        return self.x[self.num_train+self.num_val:]

    @property
    def y_test(self) -> torch.Tensor:
        return self.y[self.num_train+self.num_val:]

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            self.train_set = TensorDataset(self.x_train, self.y_train)
            self.val_set = TensorDataset(self.x_val, self.y_val)

        # create test dataset
        elif stage == 'test':
            self.test_set = TensorDataset(self.x_test, self.y_test)

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''
        if hasattr(self, 'train_set'):
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Train set has not been set')

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''
        if hasattr(self, 'val_set'):
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Val. set has not been set')

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''
        if hasattr(self, 'test_set'):
            return DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Test set has not been set')

