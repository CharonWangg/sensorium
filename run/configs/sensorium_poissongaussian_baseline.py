num_classes = 2

loss = [dict(type='PoissonGaussianLoss', reduction='sum', scale=True, loss_weight=1.0)]
# cls_loss = [dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1e6)]

model = dict(
    type='FiringRateEncoder',
    need_dataloader=True,
    backbone=dict(
        type='NeuralPredictors',
        model_name='Stacked2dCore',
        input_channels=1,
        hidden_channels=64,
        input_kern=9,
        hidden_kern=7,
        layers=4,
        gamma_input=15.5,
        skip=0,
        final_nonlinearity=True,
        bias=False,
        momentum=0.9,
        pad_input=False,
        batch_norm=True,
        hidden_dilation=1,
        laplace_padding=None,
        input_regularizer='LaplaceL2norm',
        stack=-1,
        depth_separable=True,
        linear=False,
        attention_conv=False,
        hidden_padding=None,
        use_avg_reg=False),
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
    # auxiliary_head=dict(
    #     type='BaseHead',
    #     in_channels=2,
    #     channels=None,
    #     num_classes=2,
    #     losses=cls_loss,
    # ),
    evaluation=dict(metrics=[
        dict(type='Correlation'),
        dict(type='AverageCorrelation', by='frame_image_id')
    ]))

dataset_type = 'Sensorium'
data_root = '/home/sensorium/sensorium/notebooks/data'
size = (32, 64)

albu_train_transforms = [
    dict(type='Resize', height=size[0], width=size[1], p=1.0),
]
train_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=albu_train_transforms
    ),
    dict(type='ToTensor')
]
test_pipeline = [
    dict(type='LoadImages', channels_first=False, to_RGB=True),
    dict(
        type='Albumentations',
        transforms=albu_train_transforms
    ),
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
        feature_dir=mouse,
        data_keys=data_keys,
        sampler=None,
        pipeline=train_pipeline,
    ),
    val=dict(
        type='Sensorium',
        tier='validation',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=mouse,
        data_keys=data_keys,
        sampler=None,
        pipeline=test_pipeline,
    ),
    test=dict(
        type='Sensorium',
        tier='test',
        data_root='/home/sensorium/sensorium/notebooks/data',
        feature_dir=mouse,
        data_keys=data_keys,
        sampler=None,
        pipeline=test_pipeline,
    ))

log = dict(
    project_name='sensorium',
    work_dir='/data2/charon/sensorium',
    exp_name='sensorium_poissongaussian_baseline',
    logger_interval=10,
    monitor='val_correlation',
    logger=[dict(type='comet', key='You API Key')],
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
    optimizer=dict(type='Adam', lr=9e-3),
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
    # scheduler=dict(type='CosineAnnealing',
    #                interval='step',
    #                min_lr=0.0)
)
