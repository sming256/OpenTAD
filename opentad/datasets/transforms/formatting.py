import torch
import torch.nn.functional as F
import torchvision
import scipy
import numpy as np
from collections.abc import Sequence
from einops import rearrange, reduce

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PIPELINES.register_module()
class Collect:
    def __init__(
        self,
        inputs,
        keys=[],
        meta_keys=[
            "video_name",
            "data_path",
            "fps",
            "duration",
            "snippet_stride",
            "window_start_frame",
            "resize_length",
            "window_size",
            "offset_frames",
        ],
    ):
        self.inputs = inputs
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}

        # input key
        data["inputs"] = results[self.inputs]  # [C,T]

        # AutoAugment key: gt_segments, gt_labels, masks
        for key in self.keys:
            if key == "masks" and key not in results.keys():
                results["masks"] = torch.ones(data["inputs"].shape[-1]).bool()
            data[key] = results[key]

        # meta keys
        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                if key in results.keys():
                    meta[key] = results[key]
            data["metas"] = meta

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"keys={self.keys}, meta_keys={self.meta_keys}, "


@PIPELINES.register_module()
class ConvertToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


@PIPELINES.register_module()
class Rearrange:
    def __init__(self, keys, ops, **kwargs):
        self.keys = keys
        self.ops = ops
        self.kwargs = kwargs

    def __call__(self, results):
        for key in self.keys:
            results[key] = rearrange(results[key], self.ops, **self.kwargs)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}ops={self.ops})"


@PIPELINES.register_module()
class Reduce:
    def __init__(self, keys, ops, reduction):
        self.keys = keys
        self.ops = ops
        self.reduction = reduction

    def __call__(self, results):
        for key in self.keys:
            results[key] = reduce(results[key], self.ops, reduction=self.reduction)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys}ops={self.ops})reduction={self.reduction}"


@PIPELINES.register_module()
class ResizeFeat:
    def __init__(self, tool, channel_first=False):
        self.tool = tool
        self.channel_first = channel_first

    @torch.no_grad()
    def torchvision_align(self, feat, tscale):
        # input feat shape [C,T]
        pseudo_input = feat.unsqueeze(0).unsqueeze(3)  # [1,C,T,1]
        pseudo_bbox = torch.Tensor([[0, 0, 0, 1, feat.shape[1]]])
        # output feat shape [C,tscale]
        output = torchvision.ops.roi_align(
            pseudo_input.half().double(),
            pseudo_bbox.half().double(),
            output_size=(tscale, 1),
            aligned=True,
        ).to(pseudo_input.dtype)
        output = output.squeeze(0).squeeze(-1)
        return output

    @torch.no_grad()
    def gtad_align(self, feat):
        raise "not implement yet"

    @torch.no_grad()
    def bmn_align(self, feat, tscale, num_bin=1, num_sample_bin=3, pool_type="mean"):
        feat = feat.numpy()
        C, T = feat.shape

        # x is the temporal location corresponding to each location  in feature sequence
        x = [0.5 + ii for ii in range(T)]
        f = scipy.interpolate.interp1d(x, feat, axis=1)

        video_feature = []
        zero_sample = np.zeros(num_bin * C)
        tmp_anchor_xmin = [1.0 / tscale * i for i in range(tscale)]
        tmp_anchor_xmax = [1.0 / tscale * i for i in range(1, tscale + 1)]

        num_sample = num_bin * num_sample_bin
        for idx in range(tscale):
            xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx] * T)
            xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx] * T)
            if xmax < x[0]:
                video_feature.append(zero_sample)
                continue
            if xmin > x[-1]:
                video_feature.append(zero_sample)
                continue

            plen = (xmax - xmin) / (num_sample - 1)
            x_new = [xmin + plen * ii for ii in range(num_sample)]
            y_new = f(x_new)
            y_new_pool = []
            for b in range(num_bin):
                tmp_y_new = y_new[:, num_sample_bin * b : num_sample_bin * (b + 1)]
                if pool_type == "mean":
                    tmp_y_new = np.mean(tmp_y_new, axis=1)
                elif pool_type == "max":
                    tmp_y_new = np.max(tmp_y_new, axis=1)
                y_new_pool.append(tmp_y_new)
            y_new_pool = np.stack(y_new_pool, axis=1).reshape(-1)
            # y_new_pool = np.reshape(y_new_pool, [-1])
            video_feature.append(y_new_pool)
        video_feature = np.stack(video_feature, axis=1)
        return torch.from_numpy(video_feature)

    @torch.no_grad()
    def torch_interpolate(self, feat, tscale):
        # input feat shape [C,T]
        feats = F.interpolate(feat.unsqueeze(0), size=tscale, mode="linear", align_corners=False).squeeze(0)
        return feats

    def __call__(self, results):
        assert "resize_length" in results.keys(), "should have resize_length as a key"
        tscale = results["resize_length"]

        if not self.channel_first:
            feats = results["feats"].permute(1, 0)  # [T,C] -> [C,T]
        else:
            feats = results["feats"]

        assert isinstance(feats, torch.Tensor)
        assert feats.ndim == 2  # [C,T]

        if self.tool == "torchvision_align":
            resized_feat = self.torchvision_align(feats, tscale)
        elif self.tool == "gtad_align":
            resized_feat = self.gtad_align(feats, tscale)
        elif self.tool == "bmn_align":
            resized_feat = self.bmn_align(feats, tscale)
        elif self.tool == "interpolate":
            resized_feat = self.torch_interpolate(feats, tscale)

        assert resized_feat.shape[0] == feats.shape[0]
        assert resized_feat.shape[1] == tscale

        if "gt_segments" in results.keys():
            # convert gt seconds to feature grid
            results["gt_segments"] = (results["gt_segments"] / results["duration"]).clamp(min=0.0, max=1.0)
            results["gt_segments"] *= tscale

        results["feats_len_ori"] = results["feats"].shape[1]  # for future usage
        if not self.channel_first:
            results["feats"] = resized_feat.permute(1, 0)  # [C,T] -> [T,C]
        else:
            results["feats"] = resized_feat
        return results


@PIPELINES.register_module()
class Padding:
    def __init__(self, length, pad_value=0, channel_first=False):
        self.length = length
        self.pad_value = pad_value
        self.channel_first = channel_first

    def __call__(self, results):
        assert "feats" in results.keys(), "should have feats as a key"
        assert results["feats"].ndim == 2, "feats should be 2 dim"

        if self.channel_first:
            feats = results["feats"].permute(1, 0)
        else:
            feats = results["feats"]

        feat_len = feats.shape[0]
        if feat_len < self.length:
            pad = torch.ones((self.length - feat_len, feats.shape[1])) * self.pad_value
            new_feats = torch.cat((feats, pad), dim=0)

            if self.channel_first:
                results["feats"] = new_feats.permute(1, 0)
            else:
                results["feats"] = new_feats

            pad_masks = torch.zeros(self.length - feat_len).bool()
            if "masks" in results.keys():
                results["masks"] = torch.cat((results["masks"], pad_masks), dim=0)
            else:
                results["masks"] = torch.cat((torch.ones(feat_len).bool(), pad_masks), dim=0)
        else:
            print(f"feature length {feat_len} is larger than padding length. Will be resized to {self.length}.")
            results["snippet_stride"] = results["snippet_stride"] * feat_len / self.length
            results["offset_frames"] = results["offset_frames"] * feat_len / self.length
            new_feats = F.interpolate(
                feats.permute(1, 0)[None],  # [b,c,t]
                size=self.length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            # new_feats [c,t]
            results["feats"] = new_feats if self.channel_first else new_feats.permute(1, 0)
            results["masks"] = torch.ones(self.length).bool()
        return results


@PIPELINES.register_module()
class ChannelReduction:
    """Select features along the channel dimension."""

    def __init__(self, in_channels, index):
        self.in_channels = in_channels
        self.index = index
        assert len(self.index) == 2

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        assert results["feats"].shape[1] == self.in_channels  # [T,C]

        # select the features
        results["feats"] = results["feats"][:, self.index[0] : self.index[1]]
        return results
