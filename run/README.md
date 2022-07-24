# Usage

## Code base
```
├── run  # core part of the program
│   ├── configs  # all configuration files in .py (create one as you need) 
│   ├── shallowmind  # core DL framework 
│   ├── train.py  # training script 
│   ├── train.sh  # training script (for linux)
```

## Existed configs (Guanranteed to not deprecated)
* sensorium_baseline.py # sensorium baseline with poisson loss 0.299 config
* sensorium_mice_baseline.py # sensorium baseline trained on all mice 0.305 config
* sensorium_mice_finetune_baseline.py # sensorium baseline finetuned on mouse 26872, 0.306 config
* mixnet_s.py # sensorium baseline with timm backbone mixnet_s and FCN neck 0.25 config
* sensorium_tikhonov_baseline.py # sensorium baseline with tikhonov regularization 0.288 config

```angular2html
$ cd sensorium/run && python train.py --cfg=$CFG --seed=$SEED --gpu_ids=$GPUS
```
or

```
$ cd sensorium/run && bash train.sh
```