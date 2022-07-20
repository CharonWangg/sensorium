num_classes = 2
loss = [dict(type='PoissonLoss', reduction='sum', scale=True, loss_weight=1.0)]
cls_loss = [
    dict(
        type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1000000.0)
]
model = dict(
    type='FiringRateEncoder',
    need_dataloader=True,
    backbone=dict(
        type='TimmModels',
        model_name='inception_resnet_v2',
        features_only=True,
        pretrained=True),
    head=dict(
        type='NeuralPredictors',
        model_name='FullGaussian2d',
        multiple=True,
        init_mu_range=0.3,
        bias=False,
        init_sigma=0.1,
        gamma_readout=0.0076,
        gauss_type='full',
        elu_offset=0,
        grid_mean_predictor=dict(
            type='cortex',
            input_dimensions=2,
            hidden_layers=1,
            hidden_features=30,
            final_tanh=True),
        losses=[
            dict(
                type='PoissonLoss',
                reduction='sum',
                scale=True,
                loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='BaseHead',
        in_channels=2,
        channels=None,
        num_classes=2,
        losses=[
            dict(
                type='TorchLoss',
                loss_name='CrossEntropyLoss',
                loss_weight=1000000.0)
        ]),
    evaluation=dict(metrics=[
        dict(type='Correlation'),
        dict(type='AverageCorrelation', by='frame_image_id')
    ]))
dataset_type = 'Sensorium'
data_root = '/home/sensorium/sensorium/notebooks/data'
size = (128, 256)
albu_train_transforms = [dict(type='Resize', height=128, width=256, p=1.0)]
train_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=[dict(type='Resize', height=128, width=256, p=1.0)]),
    dict(type='ToTensor')
]
test_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=[dict(type='Resize', height=128, width=256, p=1.0)]),
    dict(type='ToTensor')
]
mouse = 'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip'
data_keys = [
    'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
]
data = dict(
    train_batch_size=16,
    val_batch_size=32,
    test_batch_size=32,
    num_workers=4,
    train=dict(
        type='Sensorium',
        tier='train',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=[
            'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
        ],
        sampler=None,
        pipeline=[
            dict(type='LoadImages', channels_first=False, to_RGB=True),
            dict(
                type='Albumentations',
                transforms=[dict(type='Resize', height=128, width=256,
                                 p=1.0)]),
            dict(type='ToTensor')
        ]),
    val=dict(
        type='Sensorium',
        tier='validation',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=[
            'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
        ],
        sampler=None,
        pipeline=[
            dict(type='LoadImages', channels_first=False, to_RGB=True),
            dict(
                type='Albumentations',
                transforms=[dict(type='Resize', height=128, width=256,
                                 p=1.0)]),
            dict(type='ToTensor')
        ]),
    test=dict(
        type='Sensorium',
        tier='test',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=[
            'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
        ],
        sampler=None,
        pipeline=[
            dict(type='LoadImages', channels_first=False, to_RGB=True),
            dict(
                type='Albumentations',
                transforms=[dict(type='Resize', height=128, width=256,
                                 p=1.0)]),
            dict(type='ToTensor')
        ]))
log = dict(
    project_name='sensorium',
    work_dir='/data2/charon/sensorium',
    exp_name='official_sota_baseline_dual_head',
    logger_interval=10,
    monitor='val_correlation',
    logger=[dict(type='comet', key='Your API key')],
    checkpoint=dict(
        type='ModelCheckpoint',
        filename='{exp_name}-{val_dice:.3f}',
        top_k=1,
        mode='max',
        verbose=True,
        save_last=False),
    earlystopping=dict(
        mode='max',
        strict=False,
        patience=50,
        min_delta=0.0001,
        check_finite=True,
        verbose=True))
resume_from = None
cudnn_benchmark = True
optimization = dict(
    type='epoch',
    max_iters=200,
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005),
    scheduler=dict(type='CosineAnnealing', interval='step', min_lr=0.0))
base_name = 'inception_resnet_v2_baseline.py'
