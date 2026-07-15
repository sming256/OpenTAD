"""
Run predictions over an input video and generate a video subtitles file
containing the predictions for covenient inspections.
"""

import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import json
import torch
import tqdm
from decord import VideoReader, cpu
from pathlib import Path
from tempfile import NamedTemporaryFile
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.datasets.base import SlidingWindowDataset
from opentad.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate temporal action detection predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--gt_json", type=str, default=None, help="Optional ground truth JSON file")
    parser.add_argument("--min_score", type=float, default=0.4, help="Minimum confidence score")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def get_video_info(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_count = len(vr)
    duration = frame_count / fps
    return {"fps": fps, "frame_count": frame_count, "duration": duration}


def create_temp_annotation(video_path):
    video_input = Path(video_path)
    video_name = video_input.stem

    video_info = get_video_info(video_path)

    temp_annotations = {
        "version": "1.0",
        "database": {
            video_name: {
                "duration": video_info["duration"],
                "frame": video_info["frame_count"],
                "fps": video_info["fps"],
                "subset": "validation",
                "annotations": []
            }
        }
    }

    with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_annotations, f)
        return f.name


def load_model_and_checkpoint(config_path, checkpoint_path, device, logger):
    cfg = Config.fromfile(config_path)
    logger.info(f"Loaded config from: {config_path}")

    model = build_detector(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint is epoch {checkpoint['epoch']}")

    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        state_dict = checkpoint["state_dict_ema"]
        logger.info("Using Model EMA...")
    else:
        state_dict = checkpoint["state_dict"]

    if any(key.startswith('module.') for key in state_dict.keys()):
        logger.info("Removing 'module.' prefix from DDP checkpoint...")
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    return model, cfg


def run_inference(model, cfg, video_path, temp_ann_file, device, logger, output_path):
    video_input = Path(video_path)

    test_cfg = cfg.dataset.test.copy()
    test_cfg.data_path = str(video_input.parent)
    test_cfg.ann_file = temp_ann_file

    test_dataset = build_dataset(test_cfg)
    test_loader = build_dataloader(
        test_dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # Configuration might want to save some files!
    cfg.inference["folder"] = os.path.join(output_path, "temp")
    if not hasattr(cfg.inference, 'save_raw_prediction'):
        cfg.inference.save_raw_prediction = False


    # Set sliding window flag like done in test_engine.py
    cfg.post_processing.sliding_window = isinstance(test_dataset, SlidingWindowDataset)

    model.eval()
    model = model.to(device)

    logger.info("Running inference...")

    result_dict = {}

    for data_dict in tqdm.tqdm(test_loader):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)

        with torch.no_grad():
            results = model(
                **data_dict,
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=test_dataset.class_map,
            )
            for key, value in results.items():
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].extend(value)

    return result_dict


def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def load_ground_truth(gt_json_path, video_path: Path):
    if not gt_json_path or not os.path.exists(gt_json_path):
        return []

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    # Get the relative path of the video with respect to the JSON file and drop
    # the suffix, as that's how it's stored in the JSON.
    video_rel_path = video_path.relative_to(Path(gt_json_path).parent)
    video_rel_path = str(video_rel_path.with_suffix(""))

    if video_rel_path in gt_data.get("database", {}):
        return gt_data["database"][video_rel_path].get("annotations", [])
    return []


def generate_enhanced_subtitles(detections, gt_annotations, min_score, output_path, logger):
    all_segments = []

    num_filtered = 0
    for _, video_detections in detections.items():
        for detection in video_detections:
            if detection['score'] >= min_score:
                all_segments.append({
                    'start': detection['segment'][0],
                    'end': detection['segment'][1],
                    'type': 'PRED',
                    'label': detection['label'],
                    'score': detection['score']
                })
            else:
                num_filtered += 1

    for gt in gt_annotations:
        all_segments.append({
            'start': gt['segment'][0],
            'end': gt['segment'][1],
            'type': 'GT',
            'label': gt['label'],
            'score': None
        })

    all_segments.sort(key=lambda x: x['start'])

    srt_content = []
    for i, segment in enumerate(all_segments, 1):
        srt_content.append(str(i))

        start_srt = seconds_to_srt_time(segment['start'])
        end_srt = seconds_to_srt_time(segment['end'])
        srt_content.append(f"{start_srt} --> {end_srt}")

        if segment['type'] == 'PRED':
            subtitle_text = f"[PRED] {segment['label']} [{segment['start']:.2f}, {segment['end']:.2f}]: {segment['score']:.4f}"
        else:
            subtitle_text = f"<font color='green'>[GT] {segment['label']} [{segment['start']:.2f}, {segment['end']:.2f}]</font>"

        srt_content.append(subtitle_text)
        srt_content.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))

    pred_count = sum(1 for s in all_segments if s['type'] == 'PRED')
    gt_count = sum(1 for s in all_segments if s['type'] == 'GT')

    logger.info(f"Generated subtitles: {output_path}")
    logger.info(f"Predictions (>= {min_score}): {pred_count} ({num_filtered} filtered)")
    logger.info(f"Ground truth segments: {gt_count}")

    return output_path


def save_predictions(detections, output_path, logger):
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    logger.info(f"Saved raw predictions: {output_path}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger("predict", save_dir=args.output_dir)
    logger.info("Starting temporal action detection prediction...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Input video: {args.input_video}")
    logger.info(f"Output directory: {args.output_dir}")


    video_path = Path(args.input_video)
    video_name = video_path.stem

    temp_ann_file = create_temp_annotation(args.input_video)
    logger.info(f"Created temporary annotation file: {temp_ann_file}")

    try:
        model, cfg = load_model_and_checkpoint(args.config, args.checkpoint, args.device, logger)

        results = run_inference(
            model, cfg, args.input_video, temp_ann_file, args.device, logger, args.output_dir
        )

        gt_annotations = load_ground_truth(args.gt_json, video_path)
        if gt_annotations:
            logger.info(f"Loaded {len(gt_annotations)} ground truth annotations")

        srt_output = os.path.join(args.output_dir, f"{video_name}_detections.srt")
        generate_enhanced_subtitles(results, gt_annotations, args.min_score, srt_output, logger)

        json_output = os.path.join(args.output_dir, f"{video_name}_predictions.json")
        save_predictions(results, json_output, logger)

        logger.info("Done.")

    finally:
        if os.path.exists(temp_ann_file):
            os.unlink(temp_ann_file)
            logger.info("Cleaned up temporary annotation file")


if __name__ == "__main__":
    main()
