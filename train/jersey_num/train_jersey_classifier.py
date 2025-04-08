import os
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps

import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger

from models.convnext import ConvNext


# Set environment variables (required for MLFlow)
os.environ["MLFLOW_TRACKING_USERNAME"] = "SoccerNetGR2025"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "soccer2025"


class MultiClassJerseyDataset(Dataset):
    """
    Custom PyTorch Dataset for classifying jersey numbers from images.

    Expects a root directory with subfolders named '00' to '99', each containing respective class images.
    """

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Args:
            root_dir (str): Directory with class folders named '00' to '99'.
            transform (transforms.Compose, optional): Optional torchvision transforms.
            target_size (Tuple[int, int], optional): Target size (height, width) to resize images to.
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        
        self.classes = [f"{i:02}" for i in range(100)]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.missing_classes, self.samples = self.__get_missing_classes_and_samples()

    def __get_missing_classes_and_samples(self) -> Tuple[Set[str], List[Tuple[str, int]]]:
        """
        Scans the dataset and returns missing class directories and all valid image samples.

        Returns:
            Tuple[Set[str], List[Tuple[str, int]]]: Missing class names and image paths with labels.
        """
        missing_classes = set()
        samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                missing_classes.add(class_name)
                continue

            class_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(class_dir)
                for file in files
                if file.lower().endswith(valid_extensions) and not file.startswith('.')
            ]
            
            if not class_files:
                missing_classes.add(class_name)
                continue
            
            for file_path in class_files:
                samples.append((file_path, self.class_to_idx[class_name]))

        present_classes = set(self.classes) - missing_classes
        print(f"Found {len(samples)} samples across {len(present_classes)} classes")
        print(f"Missing classes ({len(missing_classes)}): {sorted(missing_classes)}")

        class_counts = Counter(label for _, label in samples)
        print("Class distribution:", class_counts.most_common())

        return missing_classes, samples

    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        """Pads the image to make it square using black borders."""
        w, h = img.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,
            (max_side - h) // 2,
            (max_side - w + 1) // 2,
            (max_side - h + 1) // 2
        )
        return ImageOps.expand(img, border=padding, fill=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self._pad_to_square(img)
            img = F.resize(img, self.target_size)

            if self.transform:
                img = self.transform(img)

            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros(3, *self.target_size)

        return img, label


@dataclass
class DatasetConfiguration:
    """Stores training and test dataset directory paths."""
    train_dir: str
    test_dir: str


class MultiClassJerseyModule(pl.LightningModule):
    """
    PyTorch Lightning module for training a ConvNeXt-based model on 100-class jersey dataset.
    """

    def __init__(self, data_conf: DatasetConfiguration, model_name: str = "convnextv2_nano.fcmae_ft_in1k",
                 batch_size: int = 32, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_classes = 100

        self.train_dataset = MultiClassJerseyDataset(
            data_conf.train_dir,
            target_size=(224, 224),
            transform=transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        )

        self.val_dataset = MultiClassJerseyDataset(
            data_conf.test_dir,
            target_size=(224, 224)
        )

        self.model = ConvNext(model_name, self.num_classes, learning_rate=learning_rate)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=19,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=19,
            persistent_workers=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)

        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def train_multiclass(train_dir: str, test_dir: str,
                     epochs: int = 40, batch_size: int = 32,
                     model_name: str = "convnextv2_nano.fcmae_ft_in1k") -> None:
    """
    Trains and evaluates a ConvNeXt model on the jersey dataset.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the validation/test dataset directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        model_name (str): ConvNeXt model variant to use.
    """
    data_config = DatasetConfiguration(train_dir=train_dir, test_dir=test_dir)
    model = MultiClassJerseyModule(data_config, model_name=model_name, batch_size=batch_size)

    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath="./saved_models",
        filename=f"jersey_numbers_{model_name}_best",
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max",
        verbose=True
    )

    mlf_logger = MLFlowLogger(
        experiment_name="jersey_number_recognition",
        tracking_uri="http://trackme.eidos.ai/"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=20,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlf_logger,
        default_root_dir="./saved_models",
        accelerator="auto",
        devices="auto"
    )

    trainer.fit(model, model.train_loader, model.val_loader)
    trainer.test(model, model.val_loader, ckpt_path="best")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train 100-class jersey number classifier')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k",
                        help='ConvNeXt model variant to use')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='Number of training epochs')

    args = parser.parse_args()

    train_multiclass(
        train_dir="./merged_dataset",
        test_dir="./test_0.85",
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
