from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


AFFNIST_URLS = {
    "test": "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip",
    "train": "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_and_validation.mat.zip",
}

AFFNIST_MAT_NAMES = {
    "test": "test.mat",
    "train": "training_and_validation.mat",
}


class FixedRotation:
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, image):
        return TF.rotate(image, self.degrees)


class FixedAffine:
    def __init__(
        self,
        degrees: float = 0.0,
        translate: tuple[int, int] = (0, 0),
        scale: float = 1.0,
        shear: tuple[float, float] = (0.0, 0.0),
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image):
        return TF.affine(
            image,
            angle=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=list(self.shear),
        )


class IndexedRandomRotationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        degrees: tuple[float, float],
        seed: int,
        transform=None,
    ):
        self.dataset = dataset
        self.low, self.high = degrees
        self.transform = transform
        generator = torch.Generator()
        generator.manual_seed(seed)
        angles = torch.empty(len(dataset)).uniform_(self.low, self.high, generator=generator)
        self.angles = angles.tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        image = TF.rotate(image, self.angles[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class IndexedRandomAffineDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        degrees: tuple[float, float] = (0.0, 0.0),
        translate: tuple[float, float] = (0.0, 0.0),
        scale: tuple[float, float] = (1.0, 1.0),
        shear_x: tuple[float, float] = (0.0, 0.0),
        shear_y: tuple[float, float] = (0.0, 0.0),
        seed: int = 123,
        transform=None,
    ):
        self.dataset = dataset
        self.translate = translate
        self.transform = transform
        generator = torch.Generator()
        generator.manual_seed(seed)
        size = len(dataset)
        self.angles = torch.empty(size).uniform_(degrees[0], degrees[1], generator=generator).tolist()
        self.translate_x = torch.empty(size).uniform_(-translate[0], translate[0], generator=generator).tolist()
        self.translate_y = torch.empty(size).uniform_(-translate[1], translate[1], generator=generator).tolist()
        self.scales = torch.empty(size).uniform_(scale[0], scale[1], generator=generator).tolist()
        self.shear_x = torch.empty(size).uniform_(shear_x[0], shear_x[1], generator=generator).tolist()
        self.shear_y = torch.empty(size).uniform_(shear_y[0], shear_y[1], generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def affine_params(self, idx: int) -> dict[str, float]:
        return {
            "angle_degrees": float(self.angles[idx]),
            "translate_x_frac": float(self.translate_x[idx]),
            "translate_y_frac": float(self.translate_y[idx]),
            "scale": float(self.scales[idx]),
            "shear_x_degrees": float(self.shear_x[idx]),
            "shear_y_degrees": float(self.shear_y[idx]),
        }

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        width, height = image.size
        translate = (
            int(round(self.translate_x[idx] * width)),
            int(round(self.translate_y[idx] * height)),
        )
        image = TF.affine(
            image,
            angle=self.angles[idx],
            translate=translate,
            scale=self.scales[idx],
            shear=[self.shear_x[idx], self.shear_y[idx]],
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _download_affnist(split: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{AFFNIST_MAT_NAMES[split]}.zip"
    mat_path = root / AFFNIST_MAT_NAMES[split]
    if mat_path.exists():
        return mat_path
    if not zip_path.exists():
        urllib.request.urlretrieve(AFFNIST_URLS[split], zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    if not mat_path.exists():
        candidates = list(root.rglob(AFFNIST_MAT_NAMES[split]))
        if candidates:
            candidates[0].replace(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Downloaded AffNIST archive did not contain {AFFNIST_MAT_NAMES[split]}.")
    return mat_path


def _affnist_field(payload, *names):
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
        if isinstance(payload, dict) and name in payload:
            return payload[name]
    return None


def _load_affnist_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    payload = data.get("affNISTdata", data)
    images = _affnist_field(payload, "image", "images", "X")
    labels = _affnist_field(payload, "label_int", "labels", "y")
    if images is None or labels is None:
        keys = ", ".join(sorted(key for key in data if not key.startswith("__")))
        raise ValueError(f"Could not find AffNIST image/label fields in {path}. Available keys: {keys}")

    images = np.asarray(images)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if labels.min(initial=0) == 1 and labels.max(initial=0) == 10:
        labels = labels - 1

    if images.ndim == 2:
        if images.shape[0] == 1600:
            vectors = images.T
        elif images.shape[1] == 1600:
            vectors = images
        else:
            raise ValueError(f"Expected AffNIST image matrix with one dimension of 1600, got {images.shape}.")
        images = vectors.reshape(vectors.shape[0], 40, 40)
    elif images.ndim == 3:
        if images.shape[0] == 40 and images.shape[1] == 40:
            images = images.transpose(2, 0, 1)
        elif images.shape[-2:] != (40, 40):
            raise ValueError(f"Expected AffNIST images shaped as 40x40, got {images.shape}.")
    else:
        raise ValueError(f"Unsupported AffNIST image shape: {images.shape}.")

    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"AffNIST image/label count mismatch: {images.shape[0]} images, {labels.shape[0]} labels.")

    images = np.asarray(images)
    if images.max(initial=0) <= 1.0:
        images = images * 255.0
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images, labels


class AffNISTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        download: bool = True,
        transform=None,
    ):
        if split not in AFFNIST_URLS:
            raise ValueError(f"Unsupported AffNIST split {split!r}.")
        self.root = Path(root) / "affnist"
        self.split = split
        self.transform = transform
        mat_path = self.root / AFFNIST_MAT_NAMES[split]
        if not mat_path.exists():
            if not download:
                raise FileNotFoundError(f"AffNIST file not found: {mat_path}")
            mat_path = _download_affnist(split, self.root)
        self.images, self.labels = _load_affnist_mat(mat_path)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx], mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])


def _dataset_class(name: str):
    normalized = name.lower()
    if normalized == "mnist":
        return datasets.MNIST
    if normalized in {"fashionmnist", "fashion-mnist"}:
        return datasets.FashionMNIST
    if normalized == "cifar10":
        return datasets.CIFAR10
    raise ValueError(f"Unsupported dataset '{name}'.")


def build_transforms(
    dataset: str = "mnist",
    flatten: bool = False,
    rotate_degrees: float | None = None,
    affine: dict | None = None,
):
    ops = []
    if rotate_degrees is not None:
        ops.append(FixedRotation(rotate_degrees))
    if affine is not None:
        ops.append(FixedAffine(**affine))
    ops.append(transforms.ToTensor())
    if flatten:
        ops.append(transforms.Lambda(lambda x: x.view(-1)))
    return transforms.Compose(ops)


def build_post_rotation_transforms(flatten: bool = False):
    ops = [transforms.ToTensor()]
    if flatten:
        ops.append(transforms.Lambda(lambda x: x.view(-1)))
    return transforms.Compose(ops)


def get_vision_loaders(
    dataset: str = "mnist",
    data_dir: str | Path = "data",
    batch_size: int = 128,
    flatten: bool = False,
    num_workers: int = 0,
    seed: int = 123,
    rotate_degrees: float | None = None,
    random_rotate_degrees: tuple[float, float] | None = None,
    affine: dict | None = None,
    random_affine: dict | None = None,
    rotation_seed: int | None = None,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    active_transforms = [rotate_degrees is not None, random_rotate_degrees is not None, affine is not None, random_affine is not None]
    if sum(active_transforms) > 1:
        raise ValueError("Use only one geometric transform mode.")

    Dataset = _dataset_class(dataset)
    transform = build_transforms(dataset, flatten=flatten, rotate_degrees=rotate_degrees, affine=affine)
    train_data = Dataset(str(data_dir), train=True, download=True, transform=transform)
    test_data = Dataset(str(data_dir), train=False, download=True, transform=transform)

    if random_rotate_degrees is not None or random_affine is not None:
        post_transform = build_post_rotation_transforms(flatten=flatten)
        train_base = Dataset(str(data_dir), train=True, download=True, transform=None)
        test_base = Dataset(str(data_dir), train=False, download=True, transform=None)
        base_seed = seed if rotation_seed is None else rotation_seed
        if random_rotate_degrees is not None:
            train_data = IndexedRandomRotationDataset(
                train_base,
                random_rotate_degrees,
                seed=base_seed,
                transform=post_transform,
            )
            test_data = IndexedRandomRotationDataset(
                test_base,
                random_rotate_degrees,
                seed=base_seed + 1,
                transform=post_transform,
            )
        else:
            train_data = IndexedRandomAffineDataset(
                train_base,
                **random_affine,
                seed=base_seed,
                transform=post_transform,
            )
            test_data = IndexedRandomAffineDataset(
                test_base,
                **random_affine,
                seed=base_seed + 1,
                transform=post_transform,
            )

    if train_subset is not None:
        train_data = Subset(train_data, range(min(train_subset, len(train_data))))
    if test_subset is not None:
        test_data = Subset(test_data, range(min(test_subset, len(test_data))))

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_mnist_pair_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 123,
    rotate_degrees: float | None = None,
    random_rotate_degrees: tuple[float, float] | None = None,
    affine: dict | None = None,
    random_affine: dict | None = None,
    rotation_seed: int | None = None,
    train_subset: int | None = None,
    test_subset: int | None = None,
):
    flat = get_vision_loaders(
        "mnist",
        data_dir,
        batch_size,
        flatten=True,
        num_workers=num_workers,
        seed=seed,
        rotate_degrees=rotate_degrees,
        random_rotate_degrees=random_rotate_degrees,
        affine=affine,
        random_affine=random_affine,
        rotation_seed=rotation_seed,
        train_subset=train_subset,
        test_subset=test_subset,
    )
    img = get_vision_loaders(
        "mnist",
        data_dir,
        batch_size,
        flatten=False,
        num_workers=num_workers,
        seed=seed,
        rotate_degrees=rotate_degrees,
        random_rotate_degrees=random_rotate_degrees,
        affine=affine,
        random_affine=random_affine,
        rotation_seed=rotation_seed,
        train_subset=train_subset,
        test_subset=test_subset,
    )
    return {
        "flat": {"train": flat[0], "test": flat[1]},
        "img": {"train": img[0], "test": img[1]},
    }
