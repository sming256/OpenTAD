import numpy as np
from .base import ResizeDataset, PaddingDataset, SlidingWindowDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class AnetResizeDataset(ResizeDataset):
    def get_gt(self, video_info, thresh=0.01):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0])
            gt_end = float(anno["segment"][1])
            gt_scale = (gt_end - gt_start) / float(video_info["duration"])

            if (not self.filter_gt) or (gt_scale > thresh):
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
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

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                resize_length=self.resize_length,
                sample_stride=self.sample_stride,
                # resize post process setting
                fps=-1,
                duration=float(video_info["duration"]),
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class AnetPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        # if fps is not set, use the original fps
        fps = self.fps if self.fps > 0 else float(video_info["frame"]) / float(video_info["duration"])

        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0] * fps)
            gt_end = float(anno["segment"][1] * fps)

            valid_gt = (
                (gt_end - gt_start > thresh)  # duration > thresh (eg. 0.0)
                and (gt_end - self.offset_frames > 0)  # end > 0
                and (gt_start - self.offset_frames <= float(video_info["duration"]) * fps)  # start < video_length
            )
            if (not self.filter_gt) or valid_gt:
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
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
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                # if fps is not set, use the original fps
                fps=self.fps if self.fps > 0 else float(video_info["frame"]) / float(video_info["duration"]),
                duration=float(video_info["duration"]),
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class AnetSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        # if fps is not set, use the original fps
        fps = self.fps if self.fps > 0 else float(video_info["frame"]) / float(video_info["duration"])

        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = int(anno["segment"][0] * fps)
            gt_end = int(anno["segment"][1] * fps)

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
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
                fps=self.fps if self.fps > 0 else float(video_info["frame"]) / float(video_info["duration"]),
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )
        return results
