"""
`Dataset` (pytorch) class is defined.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from methods.utils import get_root_dir


class DatasetImporter(object):
    """
    Import a dataset and store it in the instance.
    """
    def __init__(self,
                 dirname,
                 train_ratio: float,
                 data_scaling: bool,
                 n_categories: int,
                 **kwargs):
        """
        :param data_scaling
        """
        # download_ucr_datasets()
        self.n_categories = n_categories
        self.data_root = get_root_dir().joinpath("dataset")

        # fetch an entire dataset
        full_dirname = get_root_dir().joinpath('dataset', dirname)
        fnames = os.listdir(full_dirname)
        X = None
        for i, fname in enumerate(fnames):
            print(f'Data loading... {round(i / len(fnames) * 100)}%') if i % 100 == 0 else None
            x = pd.read_csv(os.path.join(full_dirname, fname), header=None).values  # (h w)

            # create n channels for n categories
            h, w = x.shape
            unique_categories = np.unique(x)
            x_new = np.zeros((self.n_categories, h, w))  # (c h w)
            for j, c in enumerate(unique_categories):
                x_new[j] = np.array(x == c, dtype=float)

            if i == 0:
                b = len(fnames)
                X = np.zeros((b, self.n_categories, h, w))  # (b c h w)
            X[i] = x_new

        # split X into X_train and X_test
        self.X_train, self.X_test = train_test_split(X, train_size=train_ratio, random_state=0)

        if data_scaling:
            min_val = np.min(self.X_train)
            max_val = np.max(self.X_train)
            self.X_train = (self.X_train - min_val) / (max_val - min_val)
            self.X_test = (self.X_test - min_val) / (max_val - min_val)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)


class GeoDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporter,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind
        self.n_categories = dataset_importer.n_categories

        if kind == "train":
            self.X = dataset_importer.X_train  # (b c h w)
        elif kind == "test":
            self.X = dataset_importer.X_test  # (b c h w)
        else:
            raise ValueError

        self._len = self.X.shape[0]

    def rbf_kernel(self, x1, x2, scale=5000, sigma=100):
        return sigma * np.exp(-1 * ((x1 - x2) ** 2) / (2 * scale))

    def gram_matrix(self, xs):
        return [[self.rbf_kernel(x1, x2) for x2 in xs] for x1 in xs]

    # def x_to_x_cond(self,
    #                 facies,
    #                 mu_alpha=50,
    #                 sigma_alpha=20,
    #                 mu_beta=0,
    #                 sigma_beta=0.5,
    #                 lambda_s=4,
    #                 vertical_only=True
    #                 ):
    #     """
    #     - Input:
    #         facies: (c h w)
    #     """
    #     height = facies.shape[1]
    #     x = np.arange(0, height)
    #     if not vertical_only:
    #         cov = self.gram_matrix(x)
    #     s = np.random.poisson(lambda_s) + 1  # number of wells
    #     # well_mat = np.zeros_like(facies)  # (c h w)
    #     well_mat = np.zeros((facies.shape[0]+1, facies.shape[1], facies.shape[2]))
    #     well_mat[self.n_categories, :, :] = np.ones_like(well_mat)[0,:,:]  # (c h w); the last channel refers to the "empty" space.
    #     for _ in range(s):
    #         alpha = np.random.normal(mu_alpha, sigma_alpha)
    #         beta = np.random.normal(mu_beta, sigma_beta)
    #         mu = alpha + beta * x
    #         if not vertical_only:
    #             residual = np.random.multivariate_normal(np.zeros(len(x)), cov)
    #             sample = mu + residual
    #         else:
    #             sample = mu
    #         inds = np.where((sample < 128) & (sample > 0))[0]
    #         x_idx = x[inds]
    #         y_idx = sample.astype(int)[inds]
    #         well_mat[:self.n_categories, x_idx, y_idx] = facies[:, x_idx, y_idx]
    #         well_mat[self.n_categories, x_idx, y_idx] = 0
    #     return well_mat

    def x_to_x_cond(self, facies):
        facies = np.flip(facies, axis=1)  # (c h w)
        x = np.argmax(facies, axis=0)  # (h w)
        x_masked = np.ones_like(x) * self.n_categories  # (h w)
        nr_of_wells = np.random.randint(1, 10)
        starting_height = np.random.uniform(10, 50)
        starting_point = np.random.randint(0, x.shape[0])
        a_list = np.random.uniform(-1, 1, nr_of_wells)
        for i in range(x_masked.shape[0]):
            for j in range(nr_of_wells):
                y = (starting_height + i + 0.5) * a_list[j] + starting_point
                if np.floor(y) >= 0 and np.floor(y) < x.shape[0]:
                    x_masked[i, int(np.floor(y))] = x[i, int(np.floor(y))]
        # well_mat = np.zeros_like(facies)
        well_mat = np.zeros((facies.shape[0] + 1, facies.shape[1], facies.shape[2]))
        well_mat[self.n_categories, :, :] = np.ones_like(well_mat)[0, :, :]  # (c h w); the last channel refers to the "empty" space.
        # for i in range(well_mat.shape[0]):
        for facies_type in np.unique(x_masked):
            well_mat[facies_type, :, :] = np.where(x_masked == facies_type, 1, 0)
            # well_mat[self.n_categories, :, :] = np.where(loc, 0, 0)
        mask_loc = (x != x_masked)
        well_mat[self.n_categories, mask_loc] = 1
        well_mat[:self.n_categories, mask_loc] = 0
        well_mat = np.flip(well_mat, axis=1).copy()
        return well_mat

    def __getitem__(self, idx):
        x = self.X[idx, :]  # (c h w)
        x_cond = self.x_to_x_cond(x)

        x = torch.from_numpy(x).float()  # (c h w)
        x_cond = torch.from_numpy(x_cond).float()  # (c h w)
        return x, x_cond

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline
    dataset_importer = DatasetImporter(get_root_dir().joinpath('dataset', 'facies_200'),
                                       train_ratio=0.8,
                                       data_scaling=True,
                                       n_categories=4)
    dataset = GeoDataset\
        ("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in data_loader:
        x, x_cond = batch
        break
    print('x.shape:', x.shape)
    print('x_cond.shape:', x_cond.shape)

    # plot
    n_samples = 4
    for b in range(n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im = axes[0].imshow(x[b].argmax(dim=0), interpolation='nearest', vmin=0, vmax=4, cmap='Accent')
        plt.colorbar(im, ax=axes[0])
        axes[0].invert_yaxis()

        im = axes[1].imshow(x_cond[b].argmax(dim=0), interpolation='nearest', vmin=0, vmax=4, cmap='Accent')
        plt.colorbar(im, ax=axes[1])
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.show()