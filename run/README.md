# Usage

## Code base
```
├── run  # core part of the program
│   ├── configs  # all configuration files in .py (create one as you need)
│   ├── shallowmind  # core DL framework
│   ├── train.py  # training script
│   ├── train.sh  # training script (for linux)
```

## Existed configs
* sensorium_baseline_dual_head.py # sensorium baseline 0.293 config
* sensorium_baseline.py # sensorium baseline 0.292 config
* sensorium_center_crop_baseline.py # sensorium baseline with center crop 0.28+ config
* sensorium_center_crop_gaussian_blur_baseline.py # sensorium baseline with center crop and gaussian blur 0.27+ config
* inception_resnet_v2_baseline.py # baseline using inception_resnet_v2 as backbone without hyperparameter tuning

```angular2html
$ cd sensorium/run && python train.py --cfg=$CFG --seed=$SEED --gpu_ids=$GPUS
```
or

```
$ cd sensorium/run && bash train.sh
```
