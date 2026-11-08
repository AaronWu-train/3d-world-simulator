# Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation

[Jimmy Wang](https://jwang771.github.io)<sup>1,2</sup>, [Yifan Chou](https://yifan-chou.github.io)<sup>3</sup>, [Kaifeng Zhang](https://kywind.github.io)<sup>1</sup>, [Yixuan Wang](https://www.yixuanwang.me/)<sup>4</sup>, [Binghao Huang](https://binghao-huang.github.io)<sup>1</sup>, [Tao Yu](https://www.linkedin.com/in/taoyumit)<sup>5</sup>, [Saleh Nabi](https://www.linkedin.com/in/saleh-nabi-877ba573)<sup>5</sup>, [Yunzhu Li](https://yunzhuli.github.io)<sup>1</sup>

<sup>1</sup>Columbia University, <sup>2</sup>UC Berkeley, <sup>3</sup>University of Michigan, <sup>4</sup>NVIDIA, <sup>5</sup>Analog Devices

[Project website](https://jwang771.github.io/3dworldsimulator/) | [Paper](https://jwang771.github.io/3dworldsimulator/paper.pdf)

**We will upload data, checkpoints, and inference code soon.**

## Environment

```bash
conda env create -f environment.yml
conda activate 3dws
```


## Data

For the local Piper cloth demonstrations, see [Amber conversion and training](amber_data/script/README.md). This experimental path uses estimated monocular depth and the supplied camera calibration, with an explicit unlabelled training task.

Put each dataset under one root, for example `data/bimanual_plug_insertion_v3/{train,val}`. The file layout is in [docs/data_format.md](docs/data_format.md).

## Training

Training has three steps. Each task's datasets, and parameters related to grid and workspace are in `tasks.py`.

### 1. Hybrid particle-voxel autoencoder

```bash
python scripts/train_vae.py --task plug --data-root data --out-dir runs/plug/vae
```

`--task` is one of `cloth`, `box`, `pusht`, `plug`, `marker`, `rope`; each task has specific set parameters in `tasks.py`, including specific voxel grid dimensions, workspace bounds, whether it has tactile sensing, among other information. We also include the ability to place object-specific weights during training, which improves the simulation fidelity. 

### 2. Latents

Dynamics model training is much faster after precomputing latents. To precompute the latents, run the following:

```bash
python scripts/export_latents.py --vae-ckpt runs/plug/vae/checkpoints/latest.pth \
  --packed-dir data/bimanual_plug_insertion_v3/train --out-dir runs/plug/latents/train
python scripts/export_latents.py --vae-ckpt runs/plug/vae/checkpoints/latest.pth \
  --packed-dir data/bimanual_plug_insertion_v3/val --out-dir runs/plug/latents/val
```


### 3. 3D Latent Dynamics

```bash
L=runs/plug/latents
python scripts/train_dynamics.py --latent-dir $L/train --val-latent-dir $L/val --out-dir runs/plug/dynamics \
  --n-history 8 --ar-schedule 1 2 3 4 5 6 7 8 9 10 --ar-schedule-epochs 50 100 150 200 250 300 350 400 450 500
```

The model predicts the next latent from the last `--n-history` latents and actions. We find that having a schedule `ar-schedule` and training using predicted latents in the history dramatically improve the performance of the model.  
