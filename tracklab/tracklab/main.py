import os
import rich.logging
import torch
import hydra
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from hydra.utils import get_original_cwd
import mlflow

from tracklab.utils import monkeypatch_hydra, progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def main(cfg: Dict[str, Any]) -> int:
    """Main entry point for the tracking pipeline.
    
    Args:
        cfg: Configuration dictionary loaded from Hydra
        
    Returns:
        int: Exit status code (0 for success)
    """
    #print(cfg['dataset']['nvid'])
    #print(cfg['modules']['bbox_detector']['cfg']['path_to_checkpoint'].split('/')[-1])
    #print(cfg)
    
    # Start the MLflow run
    experiment_id = os.environ.get('MLFLOW_EXPERIMENT_ID', '139603015997722009') # Benchmark-GameStateChallenge
    try:
        mlflow.start_run(experiment_id=experiment_id)
        log.info(f"Started MLflow run with experiment ID: {experiment_id}")
    except Exception as e:
        log.error(f"Failed to start MLflow run with experiment ID {experiment_id}: {str(e)}")
        raise 
    model_name = cfg['modules']['bbox_detector']['cfg']['path_to_checkpoint'].split('/')[-1]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_folder = hydra_cfg['runtime']['output_dir']
    # Log initial parameters
    mlflow.log_params({
        "N_VID": cfg['dataset']['nvid'],
        "MODEL_NAME": model_name,
        "split": cfg['dataset']['eval_set'],
        "load_file": cfg['state']['load_file'],
        "output_folder": run_folder,
        "entire_cfg": cfg
    })
    
    
    model_family = model_name.split(".")[0].split("_")[1]
    
    base_path = Path(get_original_cwd())
    yaml_file_path = base_path / "sn_gamestate/configs/modules/bbox_detector" / f"{model_family}.yaml"
    # Log the YAML file as an artifact
    if os.path.exists(yaml_file_path):
        mlflow.log_artifact(yaml_file_path, "YOLO-yaml")
        print(f"Uploaded YAML file: {yaml_file_path}")
    else:
        print(f"YAML file not found at: {yaml_file_path}")
    
    device = init_environment(cfg)

    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules
    for module in modules:
        if module.training_enabled:
            module.train()

    # Test tracking
    if cfg.test_tracking:
        log.info(f"Starting tracking operation on {cfg.dataset.eval_set} set.")

        # Init tracker state and tracking engine
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )

        # Run tracking and visualization
        tracking_engine.track_dataset()

        # Evaluation
        evaluate(cfg, evaluator, tracker_state)

        # Save tracker state
        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_enviroment()

    return 0


def set_sharing_strategy() -> None:
    """Set PyTorch multiprocessing sharing strategy to 'file_system'."""
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg: Dict[str, Any]) -> str:
    """Initialize the execution environment.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        str: The device string ('cuda' or 'cpu')
    """
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")
    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    return device


def close_enviroment() -> None:
    """Clean up the environment by finishing W&B run."""
    wandb.finish()


def evaluate(cfg: Dict[str, Any], evaluator: Any, tracker_state: TrackerState) -> None:
    """Run evaluation on tracking results if conditions are met.
    
    Args:
        cfg: Configuration dictionary
        evaluator: The evaluator instance
        tracker_state: Current tracker state
    """
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif cfg.get("eval_tracking", True) == False:
        log.warning("Skipping evaluation because 'eval_tracking' was set to False.")
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )


if __name__ == "__main__":
    main()