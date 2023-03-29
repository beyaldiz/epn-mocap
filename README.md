<div align="center">

# EPN-MoCap

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Marker based motion capture solving. This repository contains Pytorch Lightning implementation of [Equivariant Point Network](https://arxiv.org/abs/2103.14147) approach to mocap solving.

The code follows the structure of [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)

## Installation

Easiest way to reproduce the environment is to install dependencies through the requirements.txt and also following the installation steps in [equi-pose](https://github.com/dragonlong/equi-pose) repository:

## Usage

### Data 

Refer to [DeepMC](https://github.com/beyaldiz/DeepMC).

### Models

Model components can be found at `src/models/`.

#### [EPN Mocap Lightning model](https://github.com/beyaldiz/epn-mocap/blob/main/src/models/epn_mocap_model.py)

### Training

Training configs are set in [config.yaml](https://github.com/beyaldiz/epn-mocap/blob/main/configs/config.yaml). Once the configs are set, models can be trained as follows:
```
python run.py
```

## References
- __The code is built on the codebase of [equi-pose](https://github.com/dragonlong/equi-pose)__ and borrows codes from that repository.
- [MoCap-Solver](https://github.com/NetEase-GameAI/MoCap-Solver)
- [Equivariant Point Network](https://arxiv.org/abs/2103.14147)
