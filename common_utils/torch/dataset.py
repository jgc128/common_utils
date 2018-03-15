import numpy as np
import torch.utils.data.dataset
import torch.cuda

from sklearn.model_selection import train_test_split


class SubsetAttr(torch.utils.data.dataset.Subset):
    def __init__(self, *args, **kwargs):
        super(SubsetAttr, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __setattr__(self, key, value):
        if key == 'dataset' or key == 'indices':
            super().__setattr__(key, value)
        else:
            setattr(self.dataset, key, value)


def train_val_split(dataset, val_size=0.2, test_size=None):
    idx = np.arange(len(dataset))

    if test_size is not None:
        idx, idx_test = train_test_split(idx, test_size=test_size, random_state=0)
    else:
        idx_test = None

    idx_train, idx_val = train_test_split(idx, test_size=val_size, random_state=42)
    indices = [idx_train, idx_val]

    if idx_test is not None:
        indices.append(idx_test)

    subsets = [SubsetAttr(dataset, idxs) for idxs in indices]

    return subsets


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=1):
    pin_memory = torch.cuda.is_available()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
