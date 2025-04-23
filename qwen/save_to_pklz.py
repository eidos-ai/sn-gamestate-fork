import os
import zipfile
import pickle
import json
import argparse
import pandas as pd
from typing import Dict
import io

def update_pklz(pklz_path_in: str, pklz_path_out: str, detections_path: str):
    """
    Update a PKLZ file with jersey number detections from a JSON file.
    
    Args:
        pklz_path_in: Path to input PKLZ file
        pklz_path_out: Path to output PKLZ file
        detections_path: Path to JSON file containing detection results
    """
    with open(detections_path) as f:
        results = json.load(f)
    
    with zipfile.ZipFile(pklz_path_in, "r") as zin, \
         zipfile.ZipFile(pklz_path_out, "w", compression=zipfile.ZIP_DEFLATED) as zout:

        for name in zin.namelist():
            if name.endswith(".pkl") and not name.endswith("_image.pkl"):
                print(f"Processing {name}")
                video_id = os.path.splitext(os.path.basename(name))[0]

                # load dataframe
                with zin.open(name) as f:
                    df = pickle.load(f)

                # update jersey numbers and confidence
                if video_id in results:
                    for track_id, qwen_detection in results[video_id].items():
                        # skip non-usable detections
                        if qwen_detection.get("status") == "too_short" or "exception" in qwen_detection:
                            continue

                        # rows belonging to this track
                        mask = df["track_id"] == float(track_id)
                        track_rows_idx = df.loc[mask].index.tolist()

                        # iterate over every segment in result
                        for segment in qwen_detection.get("result", []):
                            segment_start = segment['start_frame']
                            segment_end = segment['end_frame']
                            track_length = len(track_rows_idx)
                            
                            # Fix sometimes doesnt take last frame
                            if segment_end == track_length - 1:
                                segment_end += 1
                            jersey_number = segment.get("jersey_number")
                            start_idx = int(segment_start)
                            end_idx = int(segment_end)

                            segment_indices = track_rows_idx[start_idx:end_idx]
                            
                            # Jersey_number can only be None or int string.
                            if jersey_number == "null":
                                jersey_number = None                                
                            try:
                                val = int(jersey_number)
                                if not (0 <= val <= 10000):
                                    jersey_number = None
                            except (ValueError, TypeError):
                                jersey_number = None             
                            assert jersey_number is None or 0 <= int(jersey_number) <= 10000
                            
                            df.loc[segment_indices, "jersey_number_detection"] = jersey_number
                            df.loc[segment_indices, "jersey_number_confidence"] = 0.77

                # write modified dataframe to output zip
                buf = io.BytesIO()
                pickle.dump(df, buf)
                zout.writestr(name, buf.getvalue())
            else:
                # copy other files unchanged
                with zin.open(name) as src:
                    zout.writestr(name, src.read())
    print("Processing completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Update PKLZ file with jersey number detections.')
    parser.add_argument('--input', required=True, help='Path to input PKLZ file')
    parser.add_argument('--output', required=True, help='Path to output PKLZ file')
    parser.add_argument('--detections', required=True, help='Path to JSON file containing detection results')
    
    args = parser.parse_args()
    
    update_pklz(args.input, args.output, args.detections)

if __name__ == "__main__":
    main()