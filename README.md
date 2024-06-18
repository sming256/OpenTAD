# OpenTAD: An Open-Source Temporal Action Detection Toolbox.

<p align="left">
<!-- <a href="https://arxiv.org/abs/xxx.xxx" alt="arXiv"> -->
    <!-- <img src="https://img.shields.io/badge/arXiv-xxx.xxx-b31b1b.svg?style=flat" /></a> -->
<a href="https://github.com/sming256/opentad/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a href="https://github.com/sming256/OpenTAD/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/sming256/OpenTAD?color=%23FF9600" /></a>
<a href="https://img.shields.io/github/stars/sming256/opentad" alt="arXiv">
    <img src="https://img.shields.io/github/stars/sming256/opentad" /></a>
</p>

OpenTAD is an open-source temporal action detection (TAD) toolbox based on PyTorch.


## 🥳 What's New

- A technical report of this library will be provided soon.
- **[2024/06/17]** 🔥 We rank 1st in the [Action Recognition](https://codalab.lisn.upsaclay.fr/competitions/776#results), [Action Detection](https://codalab.lisn.upsaclay.fr/competitions/707), and [Audio-Based Interaction Detection](https://codalab.lisn.upsaclay.fr/competitions/17921#results) tasks of the EPIC-KITCHENS-100 2024 Challenge, as well as 1st place in the [Moment Queries](https://eval.ai/web/challenges/challenge-page/1626/leaderboard/3913) task of the Ego4D 2024 Challenge by using OpenTAD! The technical report and code will be released soon!
- **[2024/06/14]** 🔥 We release [version v0.3](docs/en/changelog.md), which brings many new features and improvements.
- **[2024/04/17]** We release the AdaTAD, which can achieve average mAP of 42.90% on ActivityNet and 77.07% on THUMOS14.


## 📖 Major Features

- **Support SoTA TAD methods with modular design.** We decompose the TAD pipeline into different components, and implement them in a modular way. This design makes it easy to implement new methods and reproduce existing methods.
- **Support multiple TAD datasets.** We support 8 TAD datasets, including ActivityNet-1.3, THUMOS-14, HACS, Ego4D-MQ, Epic-Kitchens-100, FineAction, Multi-THUMOS, Charades datasets.
- **Support feature-based training and end-to-end training.** The feature-based training can easily be extended to end-to-end training with raw video input, and the video backbone can be easily replaced.
- **Release various pre-extracted features.** We release the feature extraction code, as well as many pre-extracted features on each dataset.

## 🌟 Model Zoo

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>One Stage</b>
      </td>
      <td>
        <b>Two Stage</b>
      </td>
      <td>
        <b>DETR</b>
      </td>
      <td>
        <b>End-to-End Training</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/actionformer">ActionFormer (ECCV'22)</a></li>
            <li><a href="configs/tridet">TriDet (CVPR'23)</a></li>
            <li><a href="configs/temporalmaxer">TemporalMaxer (arXiv'23)</a></li>
            <li><a href="configs/videomambasuite">VideoMambaSuite (arXiv'24)</a></li>
      </ul>
      </td>
      <td>
        <ul>
            <li><a href="configs/bmn">BMN (ICCV'19)</a></li>
            <li><a href="configs/gtad">GTAD (CVPR'20)</a></li>
            <li><a href="configs/tsi">TSI (ACCV'20)</a></li>
            <li><a href="configs/vsgn">VSGN (ICCV'21)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/tadtr">TadTR (TIP'22)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/afsd">AFSD (CVPR'21)</a></li>
          <li><a href="configs/tadtr">E2E-TAD (CVPR'22)</a></li>
          <li><a href="configs/etad">ETAD (CVPRW'23)</a></li>
          <li><a href="configs/re2tal">Re2TAL (CVPR'23)</a></li>
          <li><a href="configs/adatad">AdaTAD (CVPR'24)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

The detailed configs, results, and pretrained models of each method can be found in above folders.

## 🛠️ Installation

Please refer to [install.md](docs/en/install.md) for installation and data preparation.


## 🚀 Usage

Please refer to [usage.md](docs/en/usage.md) for details of training and evaluation scripts.


## 📄 Updates
Please refer to [changelog.md](docs/en/changelog.md) for update details.


## 🤝 Roadmap

All the things that need to be done in the future is in [roadmap.md](docs/en/roadmap.md).


## 🖊️ Citation

**[Acknowledgement]** This repo is inspired by [OpenMMLab](https://github.com/open-mmlab) project, and we give our thanks to their contributors.

If you think this repo is helpful, please cite us:

```bibtex
@misc{2024opentad,
    title={OpenTAD: An Open-Source Toolbox for Temporal Action Detection},
    author={Shuming Liu, Chen Zhao, Fatimah Zohra, Mattia Soldan, Carlos Hinojosa, Alejandro Pardo, Anthony Cioppa, Lama Alssum, Mengmeng Xu, Merey Ramazanova, Juan León Alcázar, Silvio Giancola, Bernard Ghanem},
    howpublished = {\url{https://github.com/sming256/opentad}},
    year={2024}
}
```

If you have any questions, please contact: `shuming.liu@kaust.edu.sa`.
