import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from mlflow.utils.mlflow_logger import MLflowLogger


class ValidateYOLO:
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
        mlflow_logger: MLflowLogger = None,  # type: ignore
    ) -> None:
        self.experiments = experiments
        self.base_model = base_model
        self.name = name
        self.eval_conf_th = eval_conf_th
        self.yaml = yaml
        self.project = project
        self.mlflow_tracking = mlflow_tracking
        self.batch_size = batch_size
        self.mlflow_logger = mlflow_logger

    def validate(self) -> None:
        experiments_dir = Path(self.experiments)

        # Load base model
        model = YOLO(self.base_model)

        # Set experiment name
        model_name = self.base_model.split("/")[-1:][0]
        name = self.name if self.name else f"{model_name}_{self.eval_conf_th}"

        # Set dataset path
        dataset_path = "../../data/YoloDataset"#"/".join(self.yaml.split("/")[:-1])

        # Initialize MLflowLogger if not provided
        if self.mlflow_logger is None:
            self.mlflow_logger = MLflowLogger(
                enable_run=self.mlflow_tracking,
                exp_name=self.project,
                run_name=name,
                run_description="Validation of YOLOv8 model",
                log_system_metrics=True,
            )

        # Log parameters
        #self.mlflow_logger.log_param("dataset_path", dataset_path)
        #self.mlflow_logger.log_param("conf", self.eval_conf_th)
        #self.mlflow_logger.log_param("batch_size", self.batch_size)
        self.mlflow_logger.log_param("Yolo model", self.base_model)
        self.mlflow_logger.set_run_tag("task", "ValYOLO")

        # Validate the model
        val_metrics = model.val(
            data=self.yaml,
            batch=self.batch_size,
            conf=self.eval_conf_th,
            project=str(experiments_dir),
        )

        results_path = Path(experiments_dir, self.mlflow_logger.get_run_id(), "val")

        # Log validation artifacts
        images_names = [
            "P_curve.png",
            "PR_curve.png",
            "R_curve.png",
        ]

        for im_name in images_names:
            im_path = results_path.joinpath(im_name)
            if im_path.exists():
                self.mlflow_logger.log_artifact(str(im_path), im_name)

        # Log precision and recall for the "person" class
        conf_matrix = val_metrics.confusion_matrix.matrix
        for class_idx, class_name in val_metrics.names.items():
            if class_name == "person":
                class_P = (
                    conf_matrix[class_idx, class_idx] / conf_matrix[class_idx, :].sum()
                )
                class_R = (
                    conf_matrix[class_idx, class_idx] / conf_matrix[:, class_idx].sum()
                )
                self.mlflow_logger.log_metric(f"{class_name}_precision", round(class_P, 2))
                self.mlflow_logger.log_metric(f"{class_name}_recall", round(class_R, 2))

        # Log other metrics (sanitize metric names)
        for metric_name, metric_value in val_metrics.results_dict.items():
            # Sanitize metric name by replacing invalid characters
            sanitized_metric_name = metric_name.replace("(", "").replace(")", "")
            self.mlflow_logger.log_metric(sanitized_metric_name, metric_value)

        # Stop the MLflow run
        self.mlflow_logger.stop_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--base_model",
        required=True,
        help="The base yolo model to use for training.",
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
        default="experiments/yolov8",
        help="Path to the directory where experiments are stored",
    )
    parser.add_argument(
        "--project",
        required=False,
        type=str,
        default="eidos/GameStateChallenge",
        help="The name of the project",
    )

    args = parser.parse_args()
    validate_yolo = ValidateYOLO(
        args.base_model,
        args.experiments,
        args.name,
        args.eval_conf_th,
        args.yaml,
        args.project,
        args.mlflow,
        args.batch_size,
    )
    validate_yolo.validate()