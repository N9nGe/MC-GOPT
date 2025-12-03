<h2 align="center">
  <b>MC-GOPT: Multi-Bin and Curriculum Extensions of GOPT for Generalizable 3D Bin Packing</b>
</h2>

MC-GOPT is an extension of the original GOPT algorithm, designed to explore more diverse and scalable training strategies for 3D bin packing. While the original GOPT framework trains policies exclusively on a fixed $10^3$ cubic container, MC-GOPT expands this setting through multi–bin-size training, curriculum-based training, and combined training regimes. Our goal is to analyze how container diversity and structured learning progressions affect policy generalization across both cubic and non-cubic packing environments.

This repository builds directly on top of the official GOPT implementation and preserves nearly identical workflows for installation, training, and evaluation to ensure full compatibility. MC-GOPT introduces new experimental configurations and training schedules while maintaining the same code structure, making it easy to reproduce baseline results and extend experiments.

the GOPT repository can be found [here](https://github.com/Xiong5Heng/GOPT).

## Installation
This code has been tested on Ubuntu 20.04 with Cuda 12.1, Python3.9 and Pytorch 2.1.0.

```
git clone https://github.com/Xiong5Heng/GOPT.git
cd GOPT

conda create -n GOPT python=3.9
conda activate GOPT

# install pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# install other dependencies
pip install -r requirements.txt
```

## Training
The dataset is generated on the fly, so you can directly train the model by running the following command. To train different models, you must create/edit the config file that determines various training and environment parameters. To reproduce our experiments, change the config file path to the one of interest.

```bash
python ts_train.py --config cfg/train_configs/config_curriculum_10.yaml --device 0 
```

## Evaluation

```bash
python ts_test.py --config cfg/test_configs/config_test_noncubic_5x8x10.yaml --device 0 --ckp /path/to/policy_step_final.pth
```

If you want to visualize the packing process of one test, you can add the `--render` flag.
```bash
python ts_test.py --config cfg/config.yaml --device 0 --ckp /path/to/policy_step_final.pth --render
```

## Demo
<!-- ![demo](./images/demo.gif) -->
<div align="center">
  <img src="./images/demo.gif" alt="A simple demo" width="400">
</div>

## License
This source code is released only for academic use. Please do not use it for commercial purposes without authorization of the author.
