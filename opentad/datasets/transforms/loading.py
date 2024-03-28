import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix=""):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        # check feat format
        if isinstance(self.feat_format, str):
            self.check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                self.check_feat_format(feat_format)

    def check_feat_format(self, feat_format):
        assert feat_format in ["npy", "npz", "pt", "csv", "pkl"], print(f"not support {feat_format}")

    def read_from_tensor(self, file_path):
        feats = torch.load(file_path).float()
        return feats

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def read_from_npz(self, file_path):
        feats = np.load(file_path)["feats"].astype(np.float32)
        return feats

    def read_from_csv(self, file_path):
        feats = pd.read_csv(file_path, dtype="float32").to_numpy()
        feats = feats.astype(np.float32)
        return feats

    def read_from_pkl(self, file_path):
        feats = pickle.load(open(file_path, "rb"))
        feats = feats.astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
            elif self.feat_format == "npz":
                feats = self.read_from_npz(file_path)
            elif feat_format == "pt":
                feats = self.read_from_tensor(file_path)
            elif feat_format == "csv":
                feats = self.read_from_csv(file_path)
            elif feat_format == "pkl":
                feats = self.read_from_pkl(file_path)
        except:
            print("Missing data:", file_path)
            exit()
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        if isinstance(results["data_path"], str):
            file_path = os.path.join(results["data_path"], f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
            feats = self.load_single_feat(file_path, self.feat_format)
        elif isinstance(results["data_path"], list):
            feats = []

            # check if the feat_format is a list
            if isinstance(self.feat_format, str):
                self.feat_format = [self.feat_format] * len(results["data_path"])

            for data_path, feat_format in zip(results["data_path"], self.feat_format):
                file_path = os.path.join(data_path, f"{self.prefix}{video_name}{self.suffix}.{feat_format}")
                feats.append(self.load_single_feat(file_path, feat_format))

            max_len = max([feat.shape[0] for feat in feats])
            for i in range(len(feats)):
                if feats[i].shape[0] != max_len:
                    # assume the first dimension is T
                    tmp_feat = F.interpolate(
                        torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                        size=max_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                    feats[i] = tmp_feat.permute(1, 0).numpy()
            feats = np.concatenate(feats, axis=1)

        # sample the feature
        sample_stride = results.get("sample_stride", 1)
        if sample_stride > 1:
            feats = feats[::sample_stride]

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"feat_format={self.feat_format}"
        return repr_str


@PIPELINES.register_module()
class SlidingWindowTrunc:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, with_mask=True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results.keys(), "should have window_size as a key"
        assert isinstance(results["feats"], torch.Tensor)
        window_size = results["window_size"]

        feats_length = results["feats"].shape[0]
        start_idx = min(results["feature_start_idx"], feats_length)
        end_idx = min(results["feature_end_idx"] + 1, feats_length)

        window_feats = results["feats"][start_idx:end_idx]
        valid_len = window_feats.shape[0]

        # if the valid window is smaller than window size, pad with -1
        if valid_len < window_size:
            pad_data = torch.zeros(window_size - valid_len, window_feats.shape[1])
            window_feats = torch.cat((window_feats, pad_data), dim=0)

        # if we need padding mask (valid is 1, pad is 0)
        if self.with_mask:
            if valid_len < window_size:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
            else:
                masks = torch.ones(window_size)
            results["masks"] = masks.bool()

        results["feats"] = window_feats.float()
        return results


@PIPELINES.register_module()
class RandomTrunc:
    """Crops features within a window such that they have a large overlap with ground truth segments.
    Withing the cropping ratio, the length is sampled."""

    def __init__(
        self,
        trunc_len,
        trunc_thresh,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False,
        pad_value=0,
        channel_first=False,
    ):
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_num_trials = max_num_trials
        self.has_action = has_action
        self.no_trunc = no_trunc
        self.pad_value = pad_value
        self.channel_first = channel_first

    def trunc_features(self, feats, gt_segments, gt_labels, offset):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = self.trunc_len
        if feat_len <= self.trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(self.max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = torch.as_tensor([st, ed], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            if self.no_trunc:
                # with at least one action and not truncating any actions
                seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
                if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                    break
            elif self.has_action:
                # with at least one action
                if seg_idx.sum().item() > 0:
                    break
            else:
                # without any constraints
                break

        feats = feats[st:ed, :]  # [T,C]
        gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def pad_features(self, feats):
        feat_len = feats.shape[0]
        if feat_len < self.trunc_len:
            feats_pad = torch.ones((self.trunc_len - feat_len,) + feats.shape[1:]) * self.pad_value
            feats = torch.cat([feats, feats_pad], dim=0)
            masks = torch.cat([torch.ones(feat_len), torch.zeros(self.trunc_len - feat_len)])
            return feats, masks
        else:
            return feats, torch.ones(feat_len)

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        offset = 0

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [C,T] -> [T,C]

        # truncate the features
        feats, gt_segments, gt_labels = self.trunc_features(
            results["feats"],
            results["gt_segments"],
            results["gt_labels"],
            offset,
        )

        # pad the features to the fixed length
        feats, masks = self.pad_features(feats)

        results["feats"] = feats.float()
        results["masks"] = masks.bool()
        results["gt_segments"] = gt_segments
        results["gt_labels"] = gt_labels

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [T,C] -> [C,T]
        return results
