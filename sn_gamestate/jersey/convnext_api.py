import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from strhub.data.module import SceneTextDataModule
from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from .convnext_detector import ConvNext2NumberDet

log = logging.getLogger(__name__)
CHECKPOINT_PATH = Path(
    os.getenv("CONVNEXT_CHECKPOINT_PATH", "/home/federico/soccernet/sn-gamestate-fork/train/jersey_num/best_model_convnextv2_base.fcmae_ft_in1k-v1.ckpt") # the path is the default, if it is set as an env variable it will take the variable
)

class ConvNext(DetectionLevelModule):
    """A detection-level module for ConvextV2 that detects jersey numbers using PARSEQ model.
    
    Attributes:
        input_columns: List of input column names required for processing.
        output_columns: List of output column names produced by the module.
        collate_fn: Function to collate data into batches.
    """
    input_columns = ["bbox_ltwh"]
    output_columns = ["has_number"]
    collate_fn = default_collate

    def __init__(self, batch_size: int, device=None, 
                 checkpoint_path: Path = CHECKPOINT_PATH, 
                 tracking_dataset: Optional[Any] = None):
        """Initialize the PARSEQ number detection module for ConvextV2.
        
        Args:
            batch_size: Number of samples per batch.
            device: it is not used but it will be sent
            checkpoint_path: Path to the PARSEQ model checkpoint.
            tracking_dataset: Optional tracking dataset reference.
        """
        super().__init__(batch_size=batch_size)

        self.model = ConvNext2NumberDet(model_path=checkpoint_path, threshold=0.75)
        self.device = device
        self.batch_size = batch_size
        log.info(f"Tracking dataset: {tracking_dataset}")

    @torch.no_grad()
    def preprocess(self, image: np.ndarray, detection: pd.Series, 
                   metadata: pd.Series) -> Dict[str, Any]:
        """Preprocess the image crop for jersey number recognition.
        
        Args:
            image: The input image as a numpy array.
            detection: Pandas Series containing detection information.
            metadata: Pandas Series containing metadata about the image.
            
        Returns:
            A dictionary containing the preprocessed data.
        """
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = Unbatchable([crop])
        return {"img": crop}

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        has_number = []
        images_np = [img.cpu().numpy() for img in batch['img']]

        # Get batch predictions
        preds = self.run_convnext_inference(images_np)

        # Convert to list
        # has_number_list.extend(preds["has_number"].tolist())
        
        for b in preds.values:
            has_number.append(b)

        # Assign to detections (assuming process is called sequentially)
        detections['has_number'] = has_number

        return detections
        
    
    @torch.no_grad()
    def run_convnext_inference(self, images_np: np.ndarray, 
                            save_dir: Path = Path('output_predictions_convnext')) -> pd.DataFrame:
        """Run Convnext inference on a list of images and save visualizations.
        
        Args:
            images_np: List of numpy array images to process.
            save_dir: Directory to save visualization results.
            
        Returns:
            DataFrame containing recognition results with columns:
            - jersey_number_detection: Recognized text
            - jersey_number_confidence: Confidence score
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "has_number": []
        }

        for i, img in enumerate(images_np):
            pred = self.model.detect(img)
                
            if pred:
                # Save visualization
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = save_dir / f"pred_{timestamp}_{i}.jpg"
            
                # try:
                #     pil_img = Image.fromarray(img)
                #     pil_img.save(output_path)
                #     print(f"✅ Successfully saved image {i} to: {output_path.resolve()}")  # Full absolute path
                # except Exception as e:
                #     print(f"❌ Failed to save image {i}: {str(e)}")
                #     #output_path = None  # Track failures if needed

            result["has_number"].append(pred)

        return pd.DataFrame(result)