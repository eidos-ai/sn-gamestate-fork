import argparse
import numpy as np
from PIL import Image, ImageDraw
import os
from math import ceil, floor
import math
import tempfile
import re
import json
from typing import Dict, List
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from loguru import logger
import torch
import time
from pathlib import Path
from IPython.display import clear_output

from annotations import load_pklz, extract_track_detections, TrackDetection
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@dataclass
class PersonSegment:
    role: str
    jersey_number: Optional[str]
    start_frame: int
    end_frame: int

@dataclass
class TrackAnalysis:
    persons: List[PersonSegment]

SYS_PROMPT = """
You are an expert vision‑language assistant specialised in analysing soccer
player tracks.  Given a grid image of bounding‑box crops covering one track:

1. Decide whether the track shows a single individual or a switch between two.
2. If a switch exists, identify the exact frame index where identity changes.
3. For each identity segment, report:
   • role: "player", "goalkeeper", or "referee".
   • jersey_number: string "NN" if visible, else null.
   • start_frame and end_frame (inclusive, zero‑based).

**Chain‑of‑Thought Tool Use**

• When unsure, first describe what's seen. Then request a zoomed‑in subsection by replying exactly:
  `<seek>start,end</seek>`  (start/end are zero‑based indices, inclusive).

• When confident, reply exactly:
    • One identity →  
      `<return>{"role":"player|goalkeeper|referee","jersey_number":"NN"|null}</return>`
    • Two identities →  
      `<return>
         {"role":"...","jersey_number":...,"start_frame":int,"end_frame":int}
         {"role":"...","jersey_number":...,"start_frame":int,"end_frame":int}
       </return>`

Only output `<seek>` or `<return>` blocks—nothing else.

ONLY DETECT JERSEY NUMBER IF SURE! 
""".strip()

def _save_temp(img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG")
    return f"file://{tmp.name}"

def render_track_detection(
    dataset_path: str,
    video_id: str,
    track: TrackDetection,
    bbox_from: int | None = None,
    bbox_to: int | None = None,
    images_per_row: int = 10,
    scale: float = 2.0,
    margin: int = 1,
) -> Image.Image:
    import os, math
    from math import floor, ceil
    from PIL import Image, ImageDraw

    img_folder = os.path.join(dataset_path, f"SNGS-{video_id}", "img1")

    frames_bboxes = sorted(zip(track.image_ids, track.bboxes), key=lambda x: x[0])

    if bbox_from is not None or bbox_to is not None:
        bbox_from = 0 if bbox_from is None else bbox_from
        bbox_to = len(frames_bboxes) - 1 if bbox_to is None else bbox_to
        
        track_length = len(track.image_ids)
        if bbox_from > track_length:
            raise Exception(f"Invalid interval, max frame is {track_length}")
            
        frames_bboxes = frames_bboxes[bbox_from : bbox_to + 1]
    

    loaded_images = []

    for frame_id, bbox in frames_bboxes:
        frame_num = str(frame_id)[-6:]
        img_path = os.path.join(img_folder, f"{frame_num}.jpg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox.tolist() if hasattr(bbox, "tolist") else bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        x1, y1 = int(floor(x)), int(floor(y))
        x2, y2 = int(ceil(x + w)), int(ceil(y + h))

        cropped = img.crop((x1, y1, x2, y2))
        if scale != 1.0:
            new_size = (int(cropped.width * scale), int(cropped.height * scale))
            cropped = cropped.resize(new_size, Image.BICUBIC)

        loaded_images.append(cropped)

    if not loaded_images:
        raise ValueError("No valid frames found for this track/interval.")

    rows = math.ceil(len(loaded_images) / images_per_row)
    row_heights = []
    col_widths = [0] * images_per_row

    for idx, img in enumerate(loaded_images):
        row, col = divmod(idx, images_per_row)
        if row == len(row_heights):
            row_heights.append(0)
        row_heights[row] = max(row_heights[row], img.height)
        col_widths[col] = max(col_widths[col], img.width)

    grid_width = sum(col_widths) + margin * (images_per_row - 1)
    grid_height = sum(row_heights) + margin * (rows - 1)
    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")

    y_offset = 0
    for row in range(rows):
        x_offset = 0
        for col in range(images_per_row):
            idx = row * images_per_row + col
            if idx >= len(loaded_images):
                break
            grid_img.paste(loaded_images[idx], (x_offset, y_offset))
            x_offset += col_widths[col] + margin
        y_offset += row_heights[row] + margin

    return grid_img

@torch.no_grad()
def qwen_track_chain_of_thought(
    dataset_path: str,
    video_id: str,
    track: TrackDetection,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    images_per_row: int = 10,
    scale: float = 1.0,
    margin: int = 1,
    max_turns: int = 8,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> Tuple[TrackAnalysis, str]:

    track_len = len(track.image_ids)
    user_prompt = (
        "Image is a grid of bbox detections for one soccer track.\n"
        "Follow the system instructions.\n"
        f"Track length is {track_len}"
    )

    seek_pat = re.compile(r"<seek>\s*(\d+)\s*[,:\-]\s*(\d+)\s*</seek>", re.S | re.I)
    ret_pat = re.compile(r"<return>(.*?)</return>", re.S | re.I)

    # ---------- first render (safe) ----------
    try:
        full_uri = _save_temp(
            render_track_detection(
                dataset_path, video_id, track, None, None,
                images_per_row, scale, margin
            )
        )
        init_content = [
            {"type": "image", "image": full_uri},
            {"type": "text", "text": user_prompt},
        ]
    except Exception as exc:
        init_content = [
            {"type": "text", "text": f"[render error] {exc}\n{user_prompt}"},
        ]

    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": init_content},
    ]

    chat_log: List[str] = []
    result: Optional[TrackAnalysis] = None

    for turn in range(1, max_turns + 1):
        prompt_text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[prompt_text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
        reply = processor.decode(
            gen_ids[inputs.input_ids.shape[-1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        log_entry = f"[Turn {turn}] {reply}"
        chat_log.append(log_entry)
        if verbose:
            print(log_entry)

        # -------------- <return> -----------------
        if (m := ret_pat.search(reply)):
            raw = m.group(1).strip()
            try:
                data = json.loads(raw)
                persons_raw = [data] if isinstance(data, dict) else data
            except json.JSONDecodeError:
                persons_raw = json.loads("[" + re.sub(r"}\s*{", "},{", raw) + "]")
            persons = [
                PersonSegment(
                    role=p["role"],
                    jersey_number=p.get("jersey_number"),
                    start_frame=p.get("start_frame", 0),
                    end_frame=p.get("end_frame", track_len - 1),
                )
                for p in persons_raw
            ]
            result = TrackAnalysis(persons=persons)
            break

        # -------------- <seek> -------------------
        if (m := seek_pat.search(reply)):
            frm_from, frm_to = map(int, m.groups())
            try:
                seek_uri = _save_temp(
                    render_track_detection(
                        dataset_path, video_id, track,
                        frm_from, frm_to,
                        images_per_row, scale, margin
                    )
                )
                msgs.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": seek_uri},
                                {"type": "text", "text": "Requested section."},
                            ],
                        },
                    ]
                )
            except Exception as exc:
                msgs.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[render error] {exc}"},
                            ],
                        },
                    ]
                )
            continue

        msgs.append({"role": "assistant", "content": reply})

    if result is None:
        raise RuntimeError("Maximum turns reached without <return>")

    return result, "\n".join(chat_log)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate Qwen analysis for soccer tracks")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--pklz_path", type=str, required=True, help="Path to the pklz file with detections")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the Qwen model")
    args = parser.parse_args()

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model = model.to('cuda') # it was generating cpu tensors.
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load detections
    detections_by_video = load_pklz(args.pklz_path)
    results = {}
    total_time = 0
    track_count = 0

    video_items = list(detections_by_video.items())
    num_videos = len(video_items)

    for video_idx, (video_id, detections) in enumerate(video_items, start=1):
        track_detections = extract_track_detections(detections)

        for i, track in enumerate(track_detections, start=1):
            clear_output(wait=True)
            print(f"Video {video_id} {video_idx}/{num_videos} - Track {i}/{len(track_detections)}")
            track_length = len(track.image_ids)

            if video_id not in results:
                results[video_id] = {}

            if track_length <= 10:
                results[video_id][str(track.track_id)] = {
                    "track_length": track_length,
                    "status": "too_short"
                }
                continue

            try:
                start_time = time.time()
                analysis, chat_log = qwen_track_chain_of_thought(
                    args.dataset_path, video_id, track, model=model, processor=processor
                )
                end_time = time.time()

                duration = end_time - start_time
                total_time += duration
                track_count += 1

                results[video_id][str(track.track_id)] = {
                    "track_length": track_length,
                    "result": [p.__dict__ for p in analysis.persons],
                    "chat_log": chat_log,
                    "duration_sec": duration,
                }

            except Exception as exc:
                logger.exception(exc)
                results[video_id][str(track.track_id)] = {
                    "track_length": track_length,
                    "exception": str(exc),
                }

            # Save intermediate results after each track
            Path(args.output_json).write_text(json.dumps(results, indent=2))

    print(f"Finished processing all tracks. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
