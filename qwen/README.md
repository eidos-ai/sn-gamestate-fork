# Jersey Number and Referee Detection with Qwen-VL

The script generate_qwen_detections.py analyzes soccer player tracks to:
- Detect jersey numbers and player roles (these roles are not used in the pipeline)
- Identify referees (these are used)
- Save detection results to JSON and create new PKLZ files with the updated Game State.

## Example Command

```bash
python generate_qwen_detections.py \
  --dataset_path="/path/to/dataset/challenge or test" \
  --input_pklz="/path/to/input_state.pklz" \
  --output_pklz="/path/to/final_output.pklz" \
  --target_video="013"
```  
 Target video can be None in which case all videos will be processed

## Working Command Example 

```bash

python generate_qwen_detections.py \
--dataset_path="/home/federico/GameStateChallenge/data/SoccerNetGS/test" \
--input_pklz="/home/federico/soccernet/sn-gamestate-fork/outputs/sn-gamestate-fork/2025-04-18/one/states/entire_test_ouryolo.pklz" \
--output_pklz="../outputs/sn-gamestate-fork/2025-04-18/one/states/completo.pklz" \
--target_video="116"