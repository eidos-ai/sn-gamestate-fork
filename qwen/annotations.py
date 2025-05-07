import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.express as px
import zipfile
import json
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import mplcursors
import json
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from typing import Optional
from loguru import logger
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import os
import math
from PIL import Image, ImageDraw
from math import ceil, floor
import os
import math
from collections import Counter
from typing import List, Tuple, Optional
import numpy as np
import math
from tqdm.notebook import tqdm



@dataclass
class TrackAnnotation:
    track_id: int
    image_ids: List[str]
    bboxes: List[List[float]]  # [x, y, w, h]
    jersey: Optional[str]
    role: Optional[str]
    team: Optional[str]

@dataclass
class TrackDetection:
    track_id: int
    image_ids: List[str]
    bboxes: List[List[float]]  # [x, y, w, h]
    jersey: Optional[str]
    role: Optional[str]
    team: Optional[str]
    
@dataclass
class Detection:
    age: float
    bbox_conf: Optional[float]
    bbox_ltwh: List[float]
    bbox_pitch: Optional[Any]
    category_id: int
    costs: Optional[Dict[str, Any]]
    # embeddings: List[List[float]]
    has_number: bool
    has_number_conf: float
    hits: int
    ignored: Optional[bool]
    image_id: str
    jersey_number: Optional[str]
    jersey_number_confidence: float
    jersey_number_detection: Optional[float]
    matched_with: Optional[Any]
    role: str
    role_confidence: Optional[float]
    role_detection: Optional[str]
    state: str
    team: Optional[str]
    team_cluster: Optional[float]
    time_since_update: float
    track_bbox_kf_ltwh: List[float]
    track_bbox_pred_kf_ltwh: Optional[List[float]]
    track_id: int
    video_id: str
    visibility_scores: List[bool]

@dataclass
class BBoxImage:
    x: int
    y: int
    x_center: float
    y_center: float
    w: int
    h: int

@dataclass
class BBoxPitch:
    x_bottom_left: float
    y_bottom_left: float
    x_bottom_right: float
    y_bottom_right: float
    x_bottom_middle: float
    y_bottom_middle: float

@dataclass
class Annotation:
    id: str
    image_id: str
    supercategory: str
    category_id: int
    attributes: Optional[Dict[str, Any]] = None
    track_id: Optional[int] = None
    bbox_image: Optional[BBoxImage] = None
    bbox_pitch: Optional[BBoxPitch] = None
    bbox_pitch_raw: Optional[BBoxPitch] = None
    lines: Optional[Dict[str, List[Dict[str, float]]]] = None
    
def dataframe_to_detections(df) -> List[Detection]:
    return [
        Detection(
            age=row['age'],
            bbox_conf=row.get('bbox_conf'),
            bbox_ltwh=row['bbox_ltwh'],
            bbox_pitch=row.get('bbox_pitch'),
            category_id=row['category_id'],
            costs=row.get('costs'),
            # embeddings=row['embeddings'],
            has_number=row['has_number'],
            has_number_conf=row['has_number_conf'],
            hits=row['hits'],
            ignored=row.get('ignored'),
            image_id=row['image_id'],
            jersey_number=row.get('jersey_number'),
            jersey_number_confidence=row['jersey_number_confidence'],
            jersey_number_detection=row.get('jersey_number_detection'),
            matched_with=row.get('matched_with'),
            role=row['role'],
            role_confidence=row.get('role_confidence'),
            role_detection=row.get('role_detection'),
            state=row['state'],
            team=row.get('team'),
            team_cluster=row.get('team_cluster'),
            time_since_update=row['time_since_update'],
            track_bbox_kf_ltwh=row['track_bbox_kf_ltwh'],
            track_bbox_pred_kf_ltwh=row.get('track_bbox_pred_kf_ltwh'),
            track_id=row['track_id'],
            video_id=row['video_id'],
            visibility_scores=row['visibility_scores']
        )
        for _, row in df.iterrows()
    ]

def load_pklz(pklz_path):
    dataframes = {}
    with zipfile.ZipFile(pklz_path, 'r') as zf:
        video_files = [f for f in zf.namelist() if f.endswith('.pkl') and not f.endswith('_image.pkl')]
        for video_file in video_files:
            video_id = os.path.splitext(video_file)[0]
            with zf.open(video_file) as f:
                df = pickle.load(f)
                dataframes[video_id] = df
                cols_key = tuple(sorted(df.columns))
    detections_by_video = {video_id: dataframe_to_detections(df) for video_id, df in dataframes.items()}
    return detections_by_video

def extract_track_detections(detections: List[Detection]) -> List[TrackDetection]:
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.track_id].append(det)

    track_detections = []
    for tid, dets in grouped.items():
        track_detections.append(
            TrackDetection(
                track_id=tid,
                image_ids=[d.image_id for d in dets],
                bboxes=[d.bbox_ltwh for d in dets],
                jersey=safe_det_jersey(dets[0].jersey_number),
                role=dets[0].role,
                team=dets[0].team,
            )
        )
    return track_detections


def extract_track_annotations(annotations: List[Annotation]) -> List[TrackAnnotation]:
    grouped = defaultdict(list)
    for ann in annotations:
        if ann.track_id is not None and ann.bbox_image:
            grouped[ann.track_id].append(ann)

    track_annotations = []
    for tid, anns in grouped.items():
        track_annotations.append(
            TrackAnnotation(
                track_id=tid,
                image_ids=[a.image_id for a in anns],
                bboxes=[[a.bbox_image.x, a.bbox_image.y, a.bbox_image.w, a.bbox_image.h] for a in anns],
                jersey=anns[0].attributes.get("jersey"),
                role=anns[0].attributes.get("role"),
                team=anns[0].attributes.get("team"),
            )
        )
    return track_annotations

def load_annotations(video_id: str, dataset_path: str) -> List[Annotation]:
    video_label_file = Path(dataset_path, f"SNGS-{video_id}", "Labels-GameState.json")
    with open(video_label_file, 'r') as f:
        data_json = json.load(f)
    
    annotations_raw = data_json['annotations']
    annotations = []
    for a in annotations_raw:
        try:
            bbox_img = a.get('bbox_image')
            bbox_pitch = a.get('bbox_pitch')
            bbox_raw = a.get('bbox_pitch_raw')

            ann = Annotation(
                id=a['id'],
                image_id=a['image_id'],
                supercategory=a['supercategory'],
                category_id=a['category_id'],
                attributes=a.get('attributes'),
                track_id=a.get('track_id'),
                bbox_image=BBoxImage(**bbox_img) if bbox_img else None,
                bbox_pitch=BBoxPitch(**bbox_pitch) if bbox_pitch else None,
                bbox_pitch_raw=BBoxPitch(**bbox_raw) if bbox_raw else None,
                lines=a.get('lines')
            )
            annotations.append(ann)
        except Exception as e:
            print(f"When processing {a}")
            logger.exception(e)
            raise e
    return annotations

def safe_det_jersey(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value_str = value.strip().lower()
        if value_str in ("", "nan"):
            return None
        return value_str
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return None
    
    
def track_coverage(
    det: TrackDetection,
    ann: TrackAnnotation,
    iou_threshold: float = 0.5
) -> float:
    matched = 0
    ann_by_frame = {frame: box.tolist() if hasattr(box, "tolist") else box for frame, box in zip(ann.image_ids, ann.bboxes)}
    det_by_frame = {frame: box.tolist() if hasattr(box, "tolist") else box for frame, box in zip(det.image_ids, det.bboxes)}

    for frame, ann_box in ann_by_frame.items():
        det_box = det_by_frame.get(frame)
        if det_box is not None and iou(det_box, ann_box) >= iou_threshold:
            matched += 1

    return matched / len(det.bboxes) if det.bboxes else 0.0


def match_tracks(
    detections: List[TrackDetection],
    annotations: List[TrackAnnotation],
    track_coverage_threshold=0.3,
    iou_threshold=0.5
) -> List[Tuple[TrackDetection, Optional[TrackAnnotation]]]:
    assignments = []
    matched_total = 0
    matched_unique = 0

    for det in detections:
        candidates = []
        for ann in annotations:
            coverage = track_coverage(det, ann, iou_threshold)
            candidates.append((ann, coverage))

        candidates.sort(key=lambda x: x[1], reverse=True)

        best_ann, best_score = candidates[0] if candidates else (None, 0.0)
        # print(f"Detection {det.track_id} best match: {best_ann.track_id if best_ann else 'None'} (Coverage: {best_score:.2f})")

        valid = [c for c in candidates if c[1] >= track_coverage_threshold]

        if len(valid) == 1:
            matched_unique += 1
            matched_total += 1
            assignments.append((det, valid[0][0]))
        elif len(valid) > 1:
            matched_total += 1
            assignments.append((det, None))  # ambiguous match
        else:
            assignments.append((det, None))  # no match

    total = len(detections)
    # print(f"\nMatched total: {matched_total}/{total} ({matched_total / total:.2%})")
    # print(f"Matched uniquely: {matched_unique}/{total} ({matched_unique / total:.2%})")
    return assignments


def iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0
    return interArea / unionArea


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
