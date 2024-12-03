import numpy as np
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=video_info["frame"] / video_info["duration"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results
