# Data Preparation for Charades

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/charades/annotations/`.


## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/charades/features/`.


We provide the following pre-extracted features for Charades:

|  Feature   |                                                Url                                                 |  Backbone  |                                                   Feature Extraction Setting                                                   |
| :--------: | :------------------------------------------------------------------------------------------------: | :--------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  vgg-rgb   | [Google Drive](https://drive.google.com/file/d/1aiRPzZHVIwGKnpxp_lMK6TOHFC93061X/view?usp=sharing) |   VGG-16   |            24fps, snippet_stride=4, converted from [official website](https://prior.allenai.org/projects/charades)             |
|  vgg-flow  | [Google Drive](https://drive.google.com/file/d/1n6KU5yYj_2btKJcr0e8bu7GrKUw2uNqE/view?usp=sharing) |   VGG-16   |            24fps, snippet_stride=4, converted from [official website](https://prior.allenai.org/projects/charades)             |
|  i3d-rgb   | [Google Drive](https://drive.google.com/file/d/1kmABW7c0wFRoYvxkYXO26iFtLU-VBf3l/view?usp=sharing) |    I3D     | 24fps, snippet_stride=8, converted from [here](https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval/blob/main/DATASET.md) |
| videomae-l | [Google Drive](https://drive.google.com/file/d/1IIPjK0mcVDY3Co3crziHV1ThvGjXuFJU/view?usp=sharing) | VideoMAE-L |                                   30fps, snippet_stride=4, clip_length=16, frame_interval=1                                    |


## Download Raw Videos

Please put the downloaded video under the path: `data/charades/raw_data/`.

You can download the raw video from [official website](https://prior.allenai.org/projects/charades), and then convert it to 30 fps.

We also provide the converted videos with 30 fps at [Google Drive](https://drive.google.com/file/d/10NiCMo5KJcTo0nCr2_hUYC1V4nzP06Eq/view?usp=sharing).

## Citation

```BibTeX
@inproceedings{sigurdsson2016hollywood,
  title={Hollywood in homes: Crowdsourcing data collection for activity understanding},
  author={Sigurdsson, Gunnar A and Varol, G{\"u}l and Wang, Xiaolong and Farhadi, Ali and Laptev, Ivan and Gupta, Abhinav},
  booktitle={Computer Vision--ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11--14, 2016, Proceedings, Part I 14},
  pages={510--526},
  year={2016},
  organization={Springer}
}
```