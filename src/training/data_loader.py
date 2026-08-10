from __future__ import annotations
from typing import Literal
from dataclasses import dataclass
from pathlib import Path
from sklearn.decomposition import PCA

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

Dataset = Literal["mnist", "fmnist"]

@dataclass(frozen=True)
class DataConfig:
    seed: int = 0
    batch_size: int = 128
    data_dir: str | Path = "data"
    val_size: int = 2_000
    flatten: bool = True
    drop_last_train: bool = True
    pca: bool = False
    pca_k: int = 784
    dataset: Dataset = "mnist"
    

def make_dataloaders(cfg: DataConfig):
    train_ds = MNIST(root=str(cfg.data_dir), train=True, download=True)
    test_ds = MNIST(root=str(cfg.data_dir), train=False, download=True)

    x_train = train_ds.data.numpy().astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int32)
    x_test = test_ds.data.numpy().astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int32)

    if cfg.flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)


    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(len(x_train))

    val_indices = indices[: cfg.val_size]
    train_indices = indices[cfg.val_size :]

    if cfg.pca:
        if cfg.pca_k < 1:
            raise ValueError("k for PCA must be positive")
        elif cfg.pca_k == 784:
            raise Warning("pca = True but pca_k = 784 make sure this is intentional")
        else:
            print(f"Fitting pca with k = {cfg.pca_k}")
            
        
        pca = PCA(
        n_components=cfg.pca_k,
        svd_solver="randomized",
        random_state=cfg.seed,
        )

        z_train = pca.fit_transform(x_train[train_indices])
        z_val = pca.transform(x_train[val_indices])
        z_test = pca.transform(x_test)

        train_data = list(zip(z_train, y_train[train_indices]))
        val_data = list(zip(z_val, y_train[val_indices]))
        test_data = list(zip(z_test, y_test))

    else:

        train_data = list(zip(x_train[train_indices], y_train[train_indices]))
        val_data = list(zip(x_train[val_indices], y_train[val_indices]))
        test_data = list(zip(x_test, y_test))
        



    torch_gen = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.drop_last_train,
        generator=torch_gen,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def make_batch(images, labels):
    return {
        "image": jnp.asarray(np.asarray(images), dtype=jnp.float32),
        "label": jnp.asarray(np.asarray(labels), dtype=jnp.int32),
    }



@dataclass(frozen=True)
class PCABasis:
    components: np.ndarray   # (k, n_pixels), orthonormal rows (no whitening)
    mean: np.ndarray         # (n_pixels,)
    k: int

    def transform(self, x_pixels: np.ndarray) -> np.ndarray:
        """Match sklearn: z = (x - mean) @ components.T."""
        x = np.asarray(x_pixels)
        return (x - self.mean) @ self.components.T


def fit_pca_basis(
    x_train_pixels: np.ndarray,
    k: int,
    *,
    seed: int,
) -> PCABasis:
    """
    Fit the PCA basis on the training pixels exactly as the data loader does
    (randomized solver, same random_state), returning a portable basis.
    """
    if k < 1:
        raise ValueError("k for PCA must be positive.")
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    pca.fit(np.asarray(x_train_pixels))
    return PCABasis(
        components=np.asarray(pca.components_),
        mean=np.asarray(pca.mean_),
        k=k,
    )