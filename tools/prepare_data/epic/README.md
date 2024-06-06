# Data Preparation for EPIC-KITCHENS-100

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/epic_kitchens-100/annotations/`.

-  You can also download original csv annotations from [official website](https://github.com/epic-kitchens/epic-kitchens-100-annotations), and use `convert_epic_kitchens_anno.py` to generate the annotations that suitable for the codebase.

## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/epic_kitchens-100/features/`.


We provide the following pre-extracted features for EPIC-KITCHENS-100:

|  Feature   |                                                                                          Url                                                                                           |              Backbone              |                 Feature Extraction Setting                 |
| :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------: | :--------------------------------------------------------: |
|  slowfast  |                                           [Google Drive](https://drive.google.com/file/d/12cDSUkJJ-id2LqKYm5fxTNDg9NOnhAAZ/view?usp=sharing)                                           |     SlowFast (Epic Finetuned)      | 30fps, snippet_stride=16, clip_length=32, frame_interval=1 |
| videomae-l | [Noun](https://drive.google.com/file/d/1YmRfMq9yn20VGifzksr5WdzUzyGCKxgF/view?usp=sharing), [Verb](https://drive.google.com/file/d/1d1snkAhErmt78GDruBN1SI176bj5fyNk/view?usp=sharing) | VideoMAE-L-16x4x1 (Epic Finetuned) | 30fps, snippet_stride=8, clip_length=16, frame_interval=1  |

## Download Raw Videos

Please put the downloaded video under the path: `data/epic_kitchens-100/raw_data/`.

You can download the raw video from [official website](https://github.com/epic-kitchens/epic-kitchens-download-scripts), then convert the videos into 30 FPS.

## Citation

```BibTeX
@ARTICLE{Damen2022RESCALING,
    title={Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
    author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria and Furnari, Antonino 
    and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
    and Perrett, Toby and Price, Will and Wray, Michael},
    journal   = {International Journal of Computer Vision (IJCV)},
    year      = {2022},
    volume = {130},
    pages = {33–55},
    Url       = {https://doi.org/10.1007/s11263-021-01531-2}
} 
```