import os
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import random
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
 

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import NeptuneLogger
import numpy as np
from PIL import Image
from PIL import ImageOps


from models.convnext import ConvNext
from models.diemodel import DieModel
from dataloader import JerseyNumberDataset
from svhn_dataloader import SVHNDataset
from types import SimpleNamespace
from synth_dataset import SnythNumberDataset

def letterbox_image(image, output_size):
    # Calculate padding size to maintain aspect ratio
    width, height = image.size
    max_dim = max(width, height)
    padding_left = (max_dim - width) // 2
    padding_right = max_dim - width - padding_left
    padding_top = (max_dim - height) // 2
    padding_bottom = max_dim - height - padding_top

    # Add padding
    image = ImageOps.expand(image, (padding_left, padding_top, padding_right, padding_bottom))

    # Resize and crop
    image = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop(output_size)(image)
    return image

@dataclass
class DatasetConfiguration:
    jerseynum_train_path: str
    jerseynum_test_path: str
    jerseynum_valid_path: str
    svhn_path: str
    synthnum_base_path: str
    train_datasets: List[str]
    valid_datasets: List[str]

    def asdict(self):
        return {
            'jerseynum_train_path': self.jerseynum_train_path,
            'jerseynum_test_path': self.jerseynum_test_path,
            'jerseynum_valid_path': self.jerseynum_valid_path,
            'svhn_path': self.svhn_path,
            'synthnum_base_path': self.jerseynum_train_path,
            'train_datasets': ','.join(self.train_datasets),
            'valid_datasets': ','.join(self.valid_datasets)
        }

    
class RandomScaledPositionTransform:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        
    def __call__(self, img):
        # Get original dimensions
        original_width, original_height = img.size
        
        # Calculate the maximum scale that keeps the image within the output size
        max_scale_width = self.output_size[0] / original_width
        max_scale_height = self.output_size[1] / original_height
        max_scale = min(max_scale_width, max_scale_height)
        
        # Ensure we only scale up to a maximum of 3, but within the bounds of the output image
        max_scale = min(max_scale, 3)
        
        # Generate a random scale between 1 and the calculated max_scale
        scale = random.uniform(1, max_scale)
        
        # Calculate the new size after scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image
        img_resized = F.resize(img, (new_height, new_width))
        
        # Calculate the maximum possible position offsets
        max_x = self.output_size[0] - new_width
        max_y = self.output_size[1] - new_height
        
        # Ensure max_x and max_y are not negative
        max_x = max(0, max_x)
        max_y = max(0, max_y)
        
        # Generate random positions within the allowed range
        left = random.randint(0, max_x)
        top = random.randint(0, max_y)
        
        # Create an empty image with the desired output size
        new_img = Image.new("RGB", self.output_size, (0, 0, 0))
        
        # Place the resized image at the random position in the new image
        new_img.paste(img_resized, (left, top))
        
        return new_img

class CenterInNewImageTransform:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def __call__(self, img):
        # Create an empty image with the desired output size
        new_img = Image.new("RGB", self.output_size, (0, 0, 0))
        
        # Calculate the coordinates to place the original image at the center
        original_width, original_height = img.size
        left = (self.output_size[0] - original_width) // 2
        top = (self.output_size[1] - original_height) // 2
        
        # Place the original image at the center of the new image
        new_img.paste(img, (left, top))
        
        return new_img
    
class DatasetRegistry():
    def __init__(self, conf: DatasetConfiguration, train_transform, inference_transform):
        # Manually define transforms here.
        # self.train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # self.inference_transform = transforms.Compose([
        #     transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(size=(224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.train_transform = train_transform
        self.inference_transform = inference_transform
        self.conf = conf
        
        self.jerseynum_train_dataset = JerseyNumberDataset(dataset_folder=conf.jerseynum_train_path, transform=self.get_transform('jerseynum_train'), include_number_not_seen=True)
        self.jersdeynum_test_dataset = JerseyNumberDataset(dataset_folder=conf.jerseynum_test_path, transform=self.get_transform('jerseynum_test'), include_number_not_seen=True)
        self.jersdeynum_val_dataset = JerseyNumberDataset(dataset_folder=conf.jerseynum_valid_path, transform=self.get_transform('jerseynum_valid'), include_number_not_seen=True)

        synth_transform = transforms.Compose([            
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.synthnum_train = SnythNumberDataset("/home/user/GameStateChallenge/data/synth", synth_transform)
        
        # SVHN images are augmented by random increase up to 3, and random positioning
        train_svhn_transform = transforms.Compose([
            RandomScaledPositionTransform(),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_svhn_transform = transforms.Compose([            
            CenterInNewImageTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_svhn =  SVHNDataset(root=conf.svhn_path, split="train", transform=train_svhn_transform)
        self.test_svhn =  SVHNDataset(root=conf.svhn_path,  split="test", transform=test_svhn_transform)
        
        self.registry = {
            'jerseynum_train': self.jerseynum_train_dataset,        
            'jerseynum_test': self.jersdeynum_test_dataset,      
            'jerseynum_valid': self.jersdeynum_val_dataset,
            'synthnum': self.synthnum_train,
            'svhn_train': self.train_svhn,
            'svhn_test': self.test_svhn,
        }
    
    def get_transform(self, dataset_name):
        is_dataset_training = dataset_name in self.conf.train_datasets
        return self.train_transform if is_dataset_training else self.inference_transform
    
    def get_train_dataset(self):
        train_datasets = ConcatDataset([self.registry[dataset] for dataset in self.conf.train_datasets])
        return train_datasets
    
    def get_valid_dataset(self):
        valid_datasets = [(self.registry[dataset], dataset) for dataset in self.conf.valid_datasets]
        return valid_datasets        

class JerseyNumTrainer(pl.LightningModule):
    def __init__(self, data_conf: DatasetConfiguration,  model_name, experiments_folder, model_path = None, batch_size = 32, learning_rate = 1e-4):       
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_conf = data_conf
        
        # First create model
        self.model = ConvNext(model_name, 100, learning_rate)
        # self.model = DieModel(model_name, 100, learning_rate)
        
        # Load model if path given
        if model_path is not None:
            self.model.load_model(model_path)
            
        self.train_transform = self.model.get_transform(is_training= True)        
        self.inference_transform = self.model.get_transform(is_training= False)
        self.data_registry = DatasetRegistry(data_conf, self.train_transform, self.inference_transform)
        
        # Then dataloaders, using model's transform
        self.train_dataset = self.data_registry.get_train_dataset() 
        self.val_dataset = self.data_registry.get_valid_dataset() 

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,  shuffle = True)        
        self.val_loaders = [DataLoader(d, batch_size=self.batch_size,  shuffle = True) for d, name in self.val_dataset]
        self.val_loaders_names = [name for d, name in self.val_dataset]
        
        self.experiment_id = None
        self.root_experiments_folder = experiments_folder
        
        self.num_classes = self.model.num_classes

        
    def get_id(self):            
        return self.experiment_id

        
    def setup(self, stage):
        # Define experiment id
        if self.logger is not None and isinstance(self.logger, NeptuneLogger):
            # If using neptune, id is that of neptune experiment
            self.experiment_id = self.logger.version
        else:
            self.experiment_id = str(uuid.uuid4())
        self.experiment_folder = Path(self.root_experiments_folder, self.experiment_id)
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        
        self.logger.experiment["training/hyperparams/model_class"] = str(self.model.__class__)
        
        self.logger.experiment["dataset/train_length"] = len(self.train_dataset)
        self.logger.experiment["dataset/val_length"] = len(self.val_dataset)
        self.logger.experiment["dataset/conf"] = self.data_conf.asdict()
        
    def _common_step(self, batch, batch_id):
        x_batch, gt_batch = batch
        pred_y_batch = self.model(x_batch)
        loss = self.model.loss(pred_y_batch, gt_batch)
        # print("pred", pred_y_batch)
        # print("gt_batch", gt_batch)
        return pred_y_batch, loss
    
    def training_step(self, batch, batch_idx):
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # print(gt_batch)
        pred_labels = self.model.pred_to_label(pred_y_batch)
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)        
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Metrics are logged to current dataloader
        dataset_name = self.val_loaders_names[dataloader_idx]
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log(f'val_loss/{dataset_name}', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx = False)
        
        pred_labels = self.model.pred_to_label(pred_y_batch)
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log(f'val_accuracy/{dataset_name}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx = False)
        
        return loss
    
    def save_model(self):
        models_folder = Path(self.experiment_folder, "models")
        models_folder.mkdir(exist_ok = True, parents = True)
        model_path = Path(models_folder, f'model_epoch_{self.current_epoch}.pt')
        torch.save(self.model.state_dict(), str(model_path))
        
    def on_validation_epoch_end(self):
        self.save_model()
        
    def configure_optimizers(self):
        return self.model.configure_optimizers()
        
def train(train_datasetpath, valid_datasetpath, test_datasetpath, svhn_datasetpath, train_datasets, valid_datasets, epochs, batch_size, model_path, model_name = "convnextv2_nano.fcmae_ft_in1k"):    

    data_configuration = DatasetConfiguration(        
        jerseynum_train_path = train_datasetpath,
        jerseynum_test_path = test_datasetpath,
        jerseynum_valid_path = valid_datasetpath,
        svhn_path = svhn_datasetpath,
        synthnum_base_path= train_datasetpath,
        train_datasets = train_datasets,
        valid_datasets = valid_datasets        
    )
    experiments_folder = "../../experiments/jersey_num"
    in_trainer = JerseyNumTrainer(data_configuration, experiments_folder=experiments_folder, batch_size=batch_size, model_name=model_name, model_path = model_path)
    
    
    neptune_logger = NeptuneLogger(  # replace with your own
        project="eidos/GameStateChallenge",  # format "workspace-name/project-name"
        tags=["TrainJerseyNumber"],  # optional
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        logger = neptune_logger,
        enable_checkpointing=False
    )
    
    # Train the Trainer
    trainer.fit(
        in_trainer, in_trainer.train_loader, in_trainer.val_loaders)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', default="../../data/jersey_num/train", required=False, help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="../../data/jersey_num/valid", required=False, help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="../../data/jersey_num/test", required=False, help='Path to directory with JerseyDataset test split')
    parser.add_argument('--svhn', default="../../data/svhn", required=False, help='Path to directory with SVHN test split')
    parser.add_argument("--train_datasets", nargs="+", type=str, help="List of training datasets")
    parser.add_argument("--valid_datasets", nargs="+", type=str, help="List of valid datasets")

    
    parser.add_argument('-ep', '--epochs', type=int, default=7, required=False, help='Number of epochs for training')
    parser.add_argument('-tb', '--batch_size', type=int, default=32, required=False, help='Batch size')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k", required=False, help='ConNext Model to use')
    parser.add_argument('-p', '--model_path', type=str, required=False, help='Path to a pretrained model to start from')
    args = vars(parser.parse_args())

    if not os.path.exists(args['train']):
        print(f"Error: Path '{args['train']}' does not exist.")
        exit(1)
    if not os.path.exists(args['val']):
        print(f"Error: Path '{args['val']}' does not exist.")
        exit(1)
    if not os.path.exists(args['test']):
        print(f"Error: Path '{args['test']}' does not exist.")
        exit(1)

    train(args['train'], args['val'], args['test'], args['svhn'],
          args['train_datasets'], args['valid_datasets'],
          epochs = args['epochs'], 
          batch_size = args['batch_size'],
          model_name = args['model_name'],
          model_path = args['model_path'])