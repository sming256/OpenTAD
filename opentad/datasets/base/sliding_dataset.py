import json
import os
import numpy as np
from mmengine.dataset import Compose

from ..builder import DATASETS, get_class_index


@DATASETS.register_module()
class SlidingWindowDataset:
    def __init__(
        self,
        ann_file,  # path of the annotation json file
        subset_name,  # name of the subset, such as training, validation, testing
        data_path,  # folder path of the raw video / pre-extracted feature
        pipeline,  # data pipeline
        class_map,  # path of the class map, convert the class id to category name
        filter_gt=False,  # if True, filter out those gt has the scale smaller than 0.01
        class_agnostic=False,  # if True, the class index will be replaced by 0
        block_list=None,  # some videos might be missed in the features or videos, we need to block them
        test_mode=False,  # if True, running on test mode with no annotation
        # for feature setting
        feature_stride=-1,  # the frames between two adjacent features, such as 4 frames
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        offset_frames=0,  # the start offset frame of the input feature
        # for sliding window setting
        window_size=-1,  # the number of features in a window
        window_overlap_ratio=0.25,  # the overlap ratio of two adjacent windows
        ioa_thresh=0.75,  # the threshold of the completeness of the gt inside the window
        fps=-1,  # some annotations are based on video-seconds
        logger=None,
    ):
        super(SlidingWindowDataset, self).__init__()

        # basic settings
        self.data_path = data_path
        self.block_list = block_list
        self.ann_file = ann_file
        self.subset_name = subset_name
        self.logger = logger.info if logger != None else print
        self.class_map = self.get_class_map(class_map)
        self.class_agnostic = class_agnostic
        self.filter_gt = filter_gt
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

        # feature settings
        self.feature_stride = int(feature_stride)
        self.sample_stride = int(sample_stride)
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = fps

        # window settings
        self.window_size = int(window_size)
        self.window_stride = int(window_size * (1 - window_overlap_ratio))
        self.ioa_thresh = ioa_thresh

        self.get_dataset()
        self.logger(
            f"{self.subset_name} subset: {len(set([data[0] for data in self.data_list]))} videos, "
            f"truncated as {len(self.data_list)} windows."
        )

    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (video_info["subset"] not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    continue

            tmp_data_list = self.split_video_to_windows(video_name, video_info, video_anno)
            self.data_list.extend(tmp_data_list)
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def split_video_to_windows(self, video_name, video_info, video_anno):
        # need: video frame, video duration, video fps
        if self.fps > 0:
            num_frames = int(video_info["duration"] * self.fps)
        else:
            num_frames = video_info["frame"]

        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False  # whether it is the last window

        for idx in range(max(1, snippet_num // self.window_stride)):  # at least one window
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:  # this is the last window
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]
            window_start_frame = window_snippet_centers[0]
            window_end_frame = window_snippet_centers[-1]

            if (video_anno != {}) and (self.ioa_thresh > 0):
                gt_segments = video_anno["gt_segments"]
                gt_labels = video_anno["gt_labels"]
                anchor = np.array([window_start_frame, window_end_frame])

                # truncate the gt segments inside the window and compute the completeness
                gt_completeness, truncated_gt = compute_gt_completeness(gt_segments, anchor)
                valid_idx = gt_completeness > self.ioa_thresh

                # only append window who has gt
                if np.sum(valid_idx) > 0:
                    window_anno = dict(
                        gt_segments=truncated_gt[valid_idx],
                        gt_labels=gt_labels[valid_idx],
                    )
                    data_list.append(
                        [
                            video_name,
                            video_info,
                            window_anno,
                            window_snippet_centers,
                        ]
                    )
            else:
                data_list.append(
                    [
                        video_name,
                        video_info,
                        video_anno,
                        window_snippet_centers,
                    ]
                )

            if last_window:  # the last window
                break

        return data_list

    def get_class_map(self, class_map_path):
        if not os.path.exists(class_map_path):
            class_map = get_class_index(self.ann_file, class_map_path)
            self.logger(f"Class map is saved in {class_map_path}, total {len(class_map)} classes.")
        else:
            with open(class_map_path, "r", encoding="utf8") as f:
                lines = f.readlines()
            class_map = [item.rstrip("\n") for item in lines]
        return class_map

    def get_gt(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.data_list)


def compute_gt_completeness(gt_boxes, anchors):
    """Compute the completeness of the gt_bboxes.
       GT will be first truncated by the anchor start/end, then the completeness is defined as the ratio of the truncated_gt_len / original_gt_len.
       If this ratio is too small, it means this gt is not complete enough to be used for training.
    Args:
        gt_boxes: np.array shape [N, 2]
        anchors:  np.array shape [2]
    """

    scores = np.zeros(gt_boxes.shape[0])  # initialized as 0
    valid_idx = np.logical_and(gt_boxes[:, 0] < anchors[1], gt_boxes[:, 1] > anchors[0])  # valid gt
    valid_gt_boxes = gt_boxes[valid_idx]

    truncated_valid_gt_len = np.minimum(valid_gt_boxes[:, 1], anchors[1]) - np.maximum(valid_gt_boxes[:, 0], anchors[0])
    original_valid_gt_len = np.maximum(valid_gt_boxes[:, 1] - valid_gt_boxes[:, 0], 1e-6)
    scores[valid_idx] = truncated_valid_gt_len / original_valid_gt_len

    # also truncated gt
    truncated_gt_boxes = np.stack(
        [np.maximum(gt_boxes[:, 0], anchors[0]), np.minimum(gt_boxes[:, 1], anchors[1])], axis=1
    )
    return scores, truncated_gt_boxes  # shape [N]
