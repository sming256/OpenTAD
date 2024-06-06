## Changelog

### v0.1.2 (2024/06/06)

New features
- OpenTAD supports temporal checkpointing now, which is proposed in ETAD and can save more memory during end-to-end training. Check [here](../../configs/etad/README.md).
- We support Ego4D-MQ dataset now. Please check [here](../../tools/prepare_data/ego4d/README.md) to download the annotation and features, and [here](../../configs/actionformer/ego4d_internvideo.py) for an implementation on ActionFormer.
- We release the config and pretrained checkpoints of AdaTAD on EPIC-KITCHENS and Ego4D datasets. Please check [here](../../configs/adatad/README.md).

Improvements
- VSGN now supports THUMOS dataset.
- ETAD now supports end-to-end training.
- The side-tuning of AdaTAD is supported now..


### v0.1.1 (2024/04/17)

- AdaTAD is released with configs / logs / pretrained checkpoints on ActivityNet dataset and THUMOS14 dataset. Please check [AdaTAD](../../configs/adatad/README.md).
- We release non-reversible version of Re2TAL, as a baseline of end-to-end actionformer. Please check [Re2TAL](../../configs/re2tal/README.md).
- Fix typos.


### v0.1.0 (2024/03/28)

- Initial release of OpenTAD.
- The codebase is under beta testing. Feedbacks and suggestions are welcome!
