# G-TAD

> [G-TAD: Sub-Graph Localization for Temporal Action Detection](https://arxiv.org/abs/1911.11462)  
> Mengmeng Xu, Chen Zhao, David S. Rojas, Ali Thabet, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Temporal action detection is a fundamental yet challenging task in video understanding. Video context is a critical cue to effectively detect actions, but current works mainly focus on temporal context, while neglecting semantic context as well as other important context properties. In this work, we propose a graph convolutional network (GCN) model to adaptively incorporate multi-level semantic context into video features and cast temporal action detection as a sub-graph localization problem. Specifically, we formulate video snippets as graph nodes, snippet-snippet correlations as edges, and actions associated with context as target sub-graphs. With graph convolution as the basic operation, we design a GCN block called GCNeXt, which learns the features of each node by aggregating its context and dynamically updates the edges in the graph. To localize each sub-graph, we also design an SGAlign layer to embed each sub-graph into the Euclidean space. Extensive experiments show that G-TAD is capable of finding effective video context without extra supervision and achieves state-of-the-art performance on two detection benchmarks. On ActivityNet-1.3, it obtains an average mAP of 34.09%; on THUMOS14, it reaches 51.6% at IoU@0.5 when combined with a proposal processing method.

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  50.19  |  35.04   |   8.11   |  34.18   | [config](anet_tsn.py) | [model](https://drive.google.com/file/d/1cyiTPZ73ud9-UxHFr7E4tkp7LEkM7_qE/view?usp=sharing)   \| [log](https://drive.google.com/file/d/10MfgZlIS8XIc0sffXnIOgOR7RIXTd5Ve/view?usp=sharing) |
|   TSP    |  52.33  |  37.58   |   8.42   |  36.20   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1CoBygy7JM26Rz7RTzIgKk0negmCa5Ier/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1ZQo2SI1TZdPkNL80FcslyForVaFDGTlB/view?usp=sharing) |


**THUMOS-14** with UtrimmedNet classifier

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSN    |  60.52  |  55.08  |  48.50  |  39.60  |  28.74  |  46.49   | [config](thumos_tsn.py) | [model](https://drive.google.com/file/d/1SCzVn2RuuIUQ3g2bktSukt0dmLR27ei0/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1VFID5LUY8p0mycYAOcjCWjNg9DkcdvEk/view?usp=sharing) |
|   I3D    |  63.35  |  59.07  |  51.76  |  42.65  |  31.66  |  49.70   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1qW80nahmt671AUR58PABzyBdQYqP9gO1/view?usp=sharing)   \| [log](https://drive.google.com/file/d/12FvCQ3j0aP4qGvus0XJaNTJRxhcxIEa5/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train GTAD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/gtad/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test GTAD on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/gtad/anet_tsp.py --checkpoint exps/anet/gtad_tsp_100x100/gpu1_id0/checkpoint/epoch_7.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@InProceedings{xu2020gtad,
  author = {Xu, Mengmeng and Zhao, Chen and Rojas, David S. and Thabet, Ali and Ghanem, Bernard},
  title = {G-TAD: Sub-Graph Localization for Temporal Action Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```