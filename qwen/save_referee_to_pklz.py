import os
import zipfile
import pickle
import json
import argparse
import pandas as pd
from typing import Dict
import io

def save_referee_to_pklz(pklz_path_in: str, pklz_path_out: str, detections_path: str, target_video_id: str|None):
    """
    Update a PKLZ file with jersey number detections from a JSON file, only for a specific video_id.
    
    Args:
        pklz_path_in: Path to input PKLZ file
        pklz_path_out: Path to output PKLZ file
        detections_path: Path to JSON file containing detection results
        target_video_id: Specific video ID to update
    """
    with open(detections_path) as f:
        results = json.load(f)
    
    with zipfile.ZipFile(pklz_path_in, "r") as zin, \
         zipfile.ZipFile(pklz_path_out, "w", compression=zipfile.ZIP_DEFLATED) as zout:

        for name in zin.namelist():
            if name.endswith(".pkl") and not name.endswith("_image.pkl"):
                video_id = os.path.splitext(os.path.basename(name))[0]
                if video_id and video_id != target_video_id:
                    # copy other files unchanged
                    with zin.open(name) as src:
                        zout.writestr(name, src.read())
                    continue

                print(f"Processing {name}")

                # load dataframe
                with zin.open(name) as f:
                    df = pickle.load(f)

                # update jersey numbers and confidence
                if video_id in results:
                    for track_id, qwen_detection in results[video_id].items():
                        if qwen_detection.get("status") == "exception" in qwen_detection:
                            continue

                        mask = df["track_id"] == float(track_id)
                        track_rows_idx = df.loc[mask].index.tolist()
                        
                        pred_referee = qwen_detection.get("result", False)
                        chat_log = qwen_detection.get("chat_log", "no chat log")
                        
                        if pred_referee:
                            df.loc[track_rows_idx, "role_detection"] = "referee"
                            df.loc[track_rows_idx, "role"] = "referee"
                            df.loc[track_rows_idx, "role_confidence"] = 1
                            df.loc[track_rows_idx, "chat_log"] = chat_log
                            
                buf = io.BytesIO()
                pickle.dump(df, buf)
                zout.writestr(name, buf.getvalue())
            else:
                with zin.open(name) as src:
                    zout.writestr(name, src.read())
    print("Processing completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Update PKLZ file with jersey number detections.')
    parser.add_argument('--input', required=True, help='Path to input PKLZ file')
    parser.add_argument('--output', required=True, help='Path to output PKLZ file')
    parser.add_argument('--detections', required=True, help='Path to JSON file containing detection results')
    parser.add_argument('--video_id', required=True, help='Video ID to process')

    args = parser.parse_args()
    
    save_referee_to_pklz(args.input, args.output, args.detections, args.video_id)

if __name__ == "__main__":
    main()
