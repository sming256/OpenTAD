import copy
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from mmengine.runner import load_checkpoint

BACKBONES = MM_BACKBONES


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized,\
                      unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print("freeze_backbone: {}, norm_eval: {}".format(self.freeze_backbone, self.norm_eval))

    def forward(self, frames, masks=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1)  # [bs*num_seg, ...]

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                features = self.model.backbone(frames)

        else:  # let the model.train() or model.eval() decide whether to freeze
            features = self.model.backbone(frames)

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False
