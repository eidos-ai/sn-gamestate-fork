from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
#from models.convnext import ConvNext
from train_jersey_detector import BinaryJerseyTrainer
import torchvision.transforms as transforms

class NumberDetector(ABC):
    """
    Abstract base class for detecting numbers in images.
    """

    @abstractmethod
    def detect(self, image) -> bool:
        """
        Detects numbers in an RGB numpy array image.

        Parameters:
            image (numpy.ndarray): The image in which to detect numbers.

        Returns:
            List[int]: List of detected numbers.
        """
        pass

class LlamaNumberDet(NumberDetector):
    """
    Detector using a LLama 3.12 Vision-Instruct.
    """

    def __init__(self):
        """
        Initializes the detector with the specified LLama 3.12 Vision-Instruct model.

        Parameters:
            
        """

        self.model_id = model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.model.tie_weights()
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        

        self.system_prompt = '''
            You are an assistant that only outputs responses in a strict JSON format. The response should contain one field: "digits". 
            Each field should be filled with the relevant information extracted from the image description. Do not include any additional comments or text outside of the JSON format. Use this format exactly:

            {
                "digits": "True if there are, False if there isn’t any",
            }

            Always ensure the response strictly follows this format and is valid JSON.
            '''

    def detect(self, image) -> bool:
        """
        Detects numbers in an image using the LLama 3.12 Vision-Instruct model.

        Parameters:
            image: RGB image to detect numbers in.

        Returns:
            bool: True if any numbers are detected, False otherwise.
        """
        user_prompt = "<|image|>\nAre there digits in this image? Give me the output in JSON format and do not add any judgments."
        full_prompt = f"SYSTEM: {self.system_prompt}\nUSER: {user_prompt}\nASSISTANT:"
        inputs = self.processor(image, full_prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs)
        full_text = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response (assuming prompt ends with "ASSISTANT:")
        assistant_response = full_text.split("ASSISTANT:")[-1].strip()
        return "True" in assistant_response
    
class LlamaNumberClassifier(NumberDetector):
    """
    Detector using a LLama 3.12 Vision-Instruct.
    """

    def __init__(self):
        """
        Initializes the detector with the specified LLama 3.12 Vision-Instruct model.

        Parameters:
            
        """

        self.model_id = model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.model.tie_weights()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.tie_weights()
        
        

        self.system_prompt = '''
            You are an assistant that only outputs responses in a strict JSON format. The response should contain one field: "number". 
            Each field should be filled with the relevant information extracted from the image description. Do not include any additional comments or text outside of the JSON format. Use this format exactly:

            {
                "number": "number you see in the image",
            }

            Always ensure the response strictly follows this format and is valid JSON.
            '''

    def detect(self, image) -> int:
        """
        Detects numbers in an image using the LLama 3.12 Vision-Instruct model.

        Parameters:
            image: RGB image to detect numbers in.

        Returns:
            int: number any numbers are detected.
        """
        user_prompt = "<|image|>\nWhat number you see in this image? Give me the output in JSON format and do not add any judgments."
        full_prompt = f"SYSTEM: {self.system_prompt}\nUSER: {user_prompt}\nASSISTANT:"
        inputs = self.processor(image, full_prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs)
        full_text = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response (assuming prompt ends with "ASSISTANT:")
        assistant_response = full_text.split("ASSISTANT:")[-1].strip()
        return full_text

class SVHNNumberDet(NumberDetector):
    """
    Detector using a model trained on the SVHN dataset.
    """

    def __init__(self, model_path: str, threshold: float):
        """
        Initializes the detector with the specified pre-trained model.

        Parameters:
            model_path (str): Path to the pre-trained model file.
        """

        self.model = YOLO(model_path)
        self.threshold = threshold

    def detect(self, image) -> bool:
        """
        Detects numbers in an image using the SVHN trained model.

        Parameters:
            image (numpy.ndarray): RGB image to detect numbers in.

        Returns:
            bool: True if any numbers are detected, False otherwise.
        """
        detections = self.model(image, verbose=False)

        #for det in detections:
        #    print(det.boxes)
        scores = [
            score
            for score in detections[0].boxes.conf.cpu().numpy().tolist()
            if score > self.threshold
        ]
        #print(f"Confidence: {max(detections[0].boxes.conf.cpu().numpy(),0)}")
        return len(scores) > 0

class ConvNext2NumberDet(NumberDetector):
    """
    Detector using a convnextv2 model.
    """

    def __init__(self, model_path: str, threshold: float):
        """
        Initializes the detector with the specified pre-trained model.

        Parameters:
            model_path (str): Path to the pre-trained model file.
        """
        self.model = BinaryJerseyTrainer.load_from_checkpoint(model_path)
        self.threshold = threshold
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, image) -> bool:
        """
        Detects numbers in an image using the SVHN trained model.

        Parameters:
            image (numpy.ndarray): RGB image to detect numbers in.

        Returns:
            bool: True if any numbers are detected, False otherwise.
        """
        trans_image = self.val_transform(image).unsqueeze(0).to(self.device)
        detections = self.model(trans_image).cpu()
        probability = torch.sigmoid(detections)

        return probability > self.threshold