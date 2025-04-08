import os
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import random
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import NeptuneLogger
import numpy as np
from PIL import Image
from models.convnext import ConvNext
from dataloader import JerseyNumberDataset
from types import SimpleNamespace

BATCH_SIZE = 8 


def calculate_accuracy_and_counts(gt: List[int], pred: List[int]) -> Tuple[List[float], List[int]]:
    """
    Calculate per-class accuracy and sample counts.
    
    Args:
        gt: List of ground truth labels
        pred: List of predicted labels
        
    Returns:
        Tuple containing:
            - List of accuracies for each class (0-99)
            - List of sample counts for each class (0-99)
    """
    class_correct = [0] * 100
    class_total = [0] * 100
    for label, prediction in zip(gt, pred):
        class_total[label] += 1
        if label == prediction:
            class_correct[label] += 1
    accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    return accuracies, class_total


def plot_heatmap(accuracies: List[float], counts: List[int], dataset_name: str) -> None:
    """
    Plot a heatmap showing accuracy and sample counts for each jersey number.
    
    Args:
        accuracies: List of accuracies for each class (0-99)
        counts: List of sample counts for each class (0-99)
        dataset_name: Name of the dataset (used for title and filename)
    """
    data = np.array(accuracies).reshape(10, 10)  # Assume accuracies are flattened from a 10x10 matrix
    sample_counts = np.array(counts).reshape(10, 10)  # Similarly for sample counts

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(data, cmap="viridis", interpolation="nearest")
    plt.colorbar(heatmap, label="Accuracy")
    plt.title(f"Accuracy and Sample Count for {dataset_name}")
    plt.xlabel("Units")
    plt.ylabel("Tens")
    plt.xticks(np.arange(10) - 0.5, range(10))
    plt.yticks(np.arange(10) - 0.5, range(10))
    plt.grid(True, linewidth=2, color="black")

    # Add text annotations for both accuracy and sample count
    for i in range(10):
        for j in range(10):
            text = f"{data[i, j]:.2f}\n({sample_counts[i, j]})"
            plt.text(j, i, text, ha="center", va="center", color="white")

    plt.savefig(f"{dataset_name}_heatmap.png")
    plt.close()
    
def save_misclassified_images(x_batch: torch.Tensor, gt_batch: torch.Tensor, 
                            pred_labels: List[int], batch_index: int, 
                            folder: str = "misclassified_images") -> None:
    """
    Save misclassified images with their ground truth and predicted labels.
    
    Args:
        x_batch: Batch of input images (tensor)
        gt_batch: Batch of ground truth labels (tensor)
        pred_labels: List of predicted labels
        batch_index: Index of the current batch
        folder: Root folder to save misclassified images
    """
    os.makedirs(folder, exist_ok=True)
    for i, (x, gt, pred) in enumerate(zip(x_batch, gt_batch, pred_labels)):
        gt = gt.item()
        if gt != pred:
            Path(folder, str(gt)).mkdir(exist_ok=True, parents=True)
            img = x.cpu().detach()  # Assuming x is a tensor
            img_path = os.path.join(folder, str(gt), f"misclassified_{batch_index}_{i}_gt{gt}_pred{pred}.png")
            save_image(img, img_path)
            annotate_image(img_path, gt, pred)
            
def annotate_image(image_path: str, gt: int, pred: int) -> None:
    """
    Annotate an image with ground truth and predicted labels.
    
    Args:
        image_path: Path to the image file
        gt: Ground truth label
        pred: Predicted label
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Using default font, adjust size if needed
    text = f"GT: {gt}, Pred: {pred}"
    draw.text((10, 10), text, font=font, fill="red")
    img.save(image_path)
    
    
def calculate_overall_accuracy(total_gt: List[int], total_pred: List[int]) -> float:
    """
    Calculate overall accuracy across all classes.
    
    Args:
        total_gt: List of all ground truth labels
        total_pred: List of all predicted labels
        
    Returns:
        Overall accuracy (correct predictions / total predictions)
    """
    correct_predictions = sum([1 for gt, pred in zip(total_gt, total_pred) if gt == pred])
    total_predictions = len(total_gt)
    return correct_predictions / total_predictions if total_predictions > 0 else 0


def remove_all(folder: str) -> None:
    """
    Remove all files and subdirectories in the given folder.
    
    Args:
        folder: Path to the folder to clean
    """
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        
def eval(train_datasetpath: str, valid_datasetpath: str, test_datasetpath: str, 
         model_path: str, model_name: str = "convnextv2_nano.fcmae_ft_in1k") -> None:
    """
    Evaluate a ConvNext model on jersey number datasets.
    
    Args:
        train_datasetpath: Path to training dataset (not currently used)
        valid_datasetpath: Path to validation dataset
        test_datasetpath: Path to test dataset
        model_path: Path to saved model weights
        model_name: Name of ConvNext model architecture
    """
    device = "cuda"
    
    model = ConvNext(model_name, 100, 1e-4)
    model.load_model(model_path)
    model.eval()
    model = model.to(device)
    inference_transform = model.get_transform(is_training=False)
    
    jerseynum_test_dataset = JerseyNumberDataset(dataset_folder=test_datasetpath, transform=inference_transform, include_number_not_seen=True)
    jerseynum_val_dataset = JerseyNumberDataset(dataset_folder=valid_datasetpath, transform=inference_transform, include_number_not_seen=True)
    
    valid_datasets = [jerseynum_test_dataset, jerseynum_val_dataset]
    dataset_names = ["jerseynum_test", "jerseynum_valid"]
    
    valid_dataloaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True) for d in valid_datasets]
    
    for dataloader, dataset_name in zip(valid_dataloaders, dataset_names):  
        # Folder to store missclassifieds
        missclassified_folder = Path("missclassified", dataset_name)
        missclassified_folder.mkdir(exist_ok=True, parents=True)
        remove_all(str(missclassified_folder))

        total_gt = []
        total_pred = []
        for batch_index, (x_batch, gt_batch) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating {dataset_name}"):
            x_batch = x_batch.to(device)
            pred_y_batch = model(x_batch)
            pred_labels = model.pred_to_label(pred_y_batch)
            # For metrics
            total_gt.extend(gt_batch.tolist())
            total_pred.extend(pred_labels.tolist())
            
            save_misclassified_images(x_batch, gt_batch, pred_labels, batch_index, missclassified_folder)

        
        overall_acc = calculate_overall_accuracy(total_gt, total_pred)
        print(f"Overall Accuracy for {dataset_name}: {overall_acc:.2f}")
        
        accuracies, counts = calculate_accuracy_and_counts(total_gt, total_pred)
        plot_heatmap(accuracies, counts, dataset_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a ConvNext model on jersey number recognition')
    parser.add_argument('-t', '--train', default="../../data/jersey_num/train", required=False, 
                       help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="../../data/jersey_num/valid", required=False, 
                       help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="../../data/jersey_num/test", required=False, 
                       help='Path to directory with JerseyDataset test split')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k", 
                       required=False, help='ConNext Model to use')
    parser.add_argument('-p', '--model_path', type=str, required=False, 
                       help='Path to a pretrained model to start from')
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

    eval(args['train'], args['val'], args['test'],         
         model_name=args['model_name'],
         model_path=args['model_path'])