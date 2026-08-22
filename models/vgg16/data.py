from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from collections import Counter
from typing import List, Tuple, Optional, Any
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import torch


class BrainMRI(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Optional[v2.Compose] = None):
        self.img_paths, self.labels = _load_images_from_root(root_dir, split)
        assert len(self.img_paths) == len(self.labels), f"{len(self.img_paths)} images mimatched with {len(self.labels)} labels"

        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = v2.ToDtype(torch.float16, scale=True)

        return img, label

    def get_class_distribution(self) -> dict:
        label_counts = Counter(self.labels)
        class_dist = {}

        for idx, class_name in enumerate(self.classes):
            class_dist[class_name] = label_counts.get(idx, 0)

        return class_dist

    def validate(self):
        warnings = []
        errors = []
        for idx in range(len(self)):
            try:
                image, label = self[idx][0], self[idx][1]
                self._validate_sample(image, label, warnings, errors)
            except Exception as e:
                errors.append(f"Exception {e} occured")

        if warnings:
            print(f"ALERT: {len(warnings)} warnings detected")
            for warning in warnings:
                print(f"    {warning}")

        if errors:
            print(f"ALERT: {len(errors)} errors detected")
            for error in errors:
                print(f"    {error}")

        if not warnings and not errors:
            print("Dataset validate - No warnings or errors detected")


    def _validate_sample(self, image: torch.Tensor, label: int, warnings: List[str], errors: List[str]) -> Tuple[List[str], List[str]]:
        assert torch.is_tensor(image), errors.append(f"Image dtype is {type(image)}")
        if torch.isnan(image).any():
            warnings.append(f"NaN values detected in image tensor")
        if torch.isinf(image).any():
            warnings.append(f"Inf values detected in image tensor")
        if label not in torch.arange(self.num_classes):
            errors.append(f"Label value {label} is invalid")

        return warnings, errors


def get_train_transforms(do_augment=False) -> v2.Compose:
    if do_augment:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True)
        ])
    
    return v2.Compose([
        v2.ToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(do_augment=False) -> v2.Compose:
    if do_augment:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True)
        ])
    
    return v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



def _load_images_from_root(root_folder_dir: str, split: str ='train') -> Tuple[List[str], List[int]]:
    """
    Builds all the image paths and their corresponding labels
    """
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    image_paths = []
    labels = []

    base_path = Path(root_folder_dir) / split

    for class_name in classes:
        class_dir = base_path / class_name

        if not class_dir.is_dir():
            print(f"ERROR: {class_name} directory does not exist")
            continue

        for path in class_dir.glob("*"):
            if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_paths.append(path)
                labels.append(class_to_idx[class_name])


    return image_paths, labels


if __name__ == "__main__":
    root_dir = Path().resolve()
    root_dir = root_dir / "data/brain-tumor-mri-deduplicated"
    print(f"Loading data from {root_dir}")

    train_dataset = BrainMRI(str(root_dir), split="train")
    print(f"Training data loaded with {len(train_dataset)} samples")

    print("\nChecking dataset validity")
    train_dataset.validate