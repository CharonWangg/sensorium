num_classes = 2
loss = [dict(type='PoissonLoss', reduction='sum', scale=True, loss_weight=1.0)]

model = dict(
    type='FiringRateEncoder',
    need_dataloader=True,
    backbone=dict(
        type='TimmModels',
        model_name='mixnet_s',
        features_only=True,
        pretrained=True),
    neck=dict(
        type='BaseNeck',
        fusion='concat',
        dim_reduction=64,
        in_channels=None,
        out_channels=None
              ),
    head=dict(
        type='NeuralPredictors',
        model_name='FullGaussian2d',
        multiple=True,
        in_index=-1,
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
        losses=loss),
    evaluation=dict(metrics=[
        dict(type='Correlation'),
        dict(type='AverageCorrelation', by='frame_image_id')
    ]))

dataset_type = 'Sensorium'
data_root = '/home/sensorium/sensorium/notebooks/data'
size = (32, 64)

albu_train_transforms = [dict(type='Resize', height=size[0], width=size[1], p=1.0),]
                         # dict(type='GaussianBlur', blur_limit=(1, 1), p=1.0)]

train_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=albu_train_transforms),
    dict(type='ToTensor')
]
test_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=albu_train_transforms),
    dict(type='ToTensor')
]
mouse = 'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip'
data_keys = [
    'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
]
data = dict(
    train_batch_size=128,
    val_batch_size=128,
    test_batch_size=128,
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
        pipeline=train_pipeline),
    val=dict(
        type='Sensorium',
        tier='validation',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=[
            'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
        ],
        pipeline=test_pipeline),
    test=dict(
        type='Sensorium',
        tier='test',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=[
            'images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'
        ],
        pipeline=test_pipeline),
    )

log = dict(
    project_name='sensorium',
    work_dir='/data2/charon/sensorium',
    exp_name='mixnet_s',
    logger_interval=10,
    monitor='val_correlation',
    logger=[dict(type='comet', key='Your API Key')],
    checkpoint=dict(
        type='ModelCheckpoint',
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
    optimizer=dict(type='Adam', lr=9e-3),
    # scheduler=dict(type='CosineAnnealing',
    #                interval='step',
    #                min_lr=0.0),
    scheduler=dict(type='ReduceLROnPlateau',
                   interval='epoch',
                   monitor='val_correlation',
                   mode="max",
                   factor=0.3,
                   patience=5,
                   threshold=1e-6,
                   min_lr=0.0001,
                   verbose=True,
                   threshold_mode="abs"),
)
