import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from mlflow.utils.mlflow_logger import MLflowLogger  # Import Eidos MLflow wrapper
from validate_yolov8 import ValidateYOLO  # type: ignore


class TrainYOLO:
    def __init__(
        self,
        base_model: str,
        experiments: str,
        name: str,
        eval_conf_th: float,
        yaml: str,
        project: str,
        mlflow_tracking: bool,
        batch_size: int,
        epochs: int,
        initial_learning_rate: float,
        final_learning_rate: float,
    ) -> None:
        self.base_model = base_model
        self.experiments = experiments
        self.name = name
        self.eval_conf_th = eval_conf_th
        self.yaml = yaml
        self.project = project
        self.mlflow_tracking = mlflow_tracking
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate

    def train(self) -> None:
        experiments_dir = Path(self.experiments)
    
        # Load base model
        model = YOLO(self.base_model)  # load a pretrained model (recommended for training)

        name = self.name if self.name else f"{self.base_model}_{self.epochs}_{self.eval_conf_th}"
    
        # dataset path
        dataset_path = "../../data/YoloDataset"  # "/".join(self.yaml.split("/")[:-1])
    
        # Initialize MLflowLogger (Eidos wrapper)
        if self.mlflow_tracking:
            mlflow_logger = MLflowLogger(
                enable_run=True,  # Enable MLflow tracking
                exp_name=self.project,  # Experiment name
                run_name=name,  # Run name
                run_description="Training YOLOv8 model",  # Run description
                log_system_metrics=True  # Log system metrics
            )
            mlflow_logger.log_param("dataset_path", dataset_path)
            mlflow_logger.log_param("epochs", self.epochs)
            mlflow_logger.log_param("eval_conf_th", self.eval_conf_th)
            mlflow_logger.log_param("batch_size", self.batch_size)
            mlflow_logger.log_param("yolo_model", self.base_model)
            mlflow_logger.log_param("initial_learning_rate", self.initial_learning_rate)
            mlflow_logger.log_param("final_learning_rate", self.final_learning_rate)
            mlflow_logger.set_run_tag("task", "TrainYOLO")
        
        print(f"MLflow Run ID: {mlflow_logger.get_run_id()}")

        # Start training
        model.train(
            data=self.yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            name=name,
            project=str(experiments_dir),
            lr0=self.initial_learning_rate,
            lrf=self.final_learning_rate,
            
        )  # train the model
        results_path = Path(experiments_dir, name)
    
        # Upload confusion matrix and other files
        if self.mlflow_tracking:
            images_names = [
                "confusion_matrix.png",
                "F1_curve.png",
                "labels.jpg",
                "P_curve.png",
                "PR_curve.png",
                "R_curve.png",
                "results.png",
            ]
            for im_name in images_names:
                im_path = results_path.joinpath(im_name)
                if im_path.exists():
                    mlflow_logger.log_artifact(str(im_path), im_name)
    
        # Perform evaluation on validation dataset
        validate_yolo = ValidateYOLO(
            self.base_model,
            self.experiments,
            self.name,
            self.eval_conf_th,
            self.yaml,
            self.project,
            self.mlflow_tracking,
            self.batch_size,
            mlflow_logger
        )
        validate_yolo.validate()
    
        # Stop MLflowLogger instance
        if self.mlflow_tracking:
            mlflow_logger.stop_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--base_model",
        required=True,
        help="The base yolo model to use for training",
    )
    parser.add_argument(
        "-y",
        "--yaml",
        required=True,
        help="Path to the yaml file containing class names and paths to train and val datasets",
    )
    parser.add_argument(
        "--name", required=False, type=str, help="The name of the experiment run"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=16,
        help="training batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        required=False,
        type=int,
        default=40,
        help="training num epochs",
    )
    parser.add_argument(
        "-c",
        "--eval_conf_th",
        required=False,
        type=float,
        default=0.5,
        help="Confidence score used for evaluation of model",
    )
    parser.add_argument(
        "--mlflow",
        required=False,
        action="store_true",
        default=False,
        help="Log experiment results to MLflow",
    )
    parser.add_argument(
        "--experiments",
        required=False,
        type=str,
        default="hola",
        help="Path to the directory where experiments are stored",
    )
    parser.add_argument(
        "--project",
        required=False,
        type=str,
        default="eidos/GameStateChallenge",
        help="The name of the project",
    )
    parser.add_argument(
        "--initial-learning-rate",
        required=False,
        type=float,
        default=0.01,
        help="The initial learning rate of the model",
    )
    parser.add_argument(
        "--final-learning-rate",
        required=False,
        type=float,
        default=0.01,
        help="The final learning rate of the model",
    )

    args = parser.parse_args()

    training = TrainYOLO(
        args.base_model,
        args.experiments,
        args.name,
        args.eval_conf_th,
        args.yaml,
        args.project,
        args.mlflow,
        args.batch_size,
        args.epochs,
        args.initial_learning_rate,
        args.final_learning_rate,
    )
    training.train()