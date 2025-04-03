# Jersey Number Dataset Generator
This script generates a dataset of cropped images featuring jersey numbers from a provided input dataset <br />
relying on available annotations and a pretrained SVHN model for number detection.

## Prerequiste
* Pretrained SVHN model for number detection

## Run the script

```
python generate.py <input_path> <model_path> [--output_path <output_path>] [--threshold <threshold>] [--cooldown_frames <cooldown_frames>]
```

* <input_path>: Path to the input dataset directory.
* <model_path>: Path to the SVHN detector model.
* [--output_path] (optional): Path to the output directory. Default is ./dataset.
* [--threshold] (optional): Threshold for SVHN detection. Default is 0.55.
* [--cooldown_frames] (optional): Cooldown frames for player detection. Default is 5.

## Example run

```
INPUT_PATH='../../data/SoccerNetGS/train'
MODEL_PATH='svhn_trained.pt'
OUTPUT_PATH='train'

python generate.py $INPUT_PATH $MODEL_PATH --output_path=$OUTPUT_PATH
```

# Jersey Number Heatmap

You can generate a heatmap visualization using the script *heatmap.py*. <br />
The script takes the dataset generated from the previous script and produces a heatmap indicating the distribution of jersey numbers within the dataset.

## Run the script

```
python heatmap.py <dataset_path> [--name <output_name>]
```

* <dataset_path>: Path to the dataset.
* [--name] (optional): Output file name for the heatmap. Default is stats.png.

## Example run

```
DATASET_PATH='train'
OUTPUT_NAME='train_cd5.png'

python heatmap.py $DATASET_PATH --name $OUTPUT_NAME
```