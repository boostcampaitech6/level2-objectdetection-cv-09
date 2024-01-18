_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/trash_detection_pseudo.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.5,
        layer_scale_init_value=0.,
        gap_before_final_norm=False,
        use_grn=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[192, 384, 768, 1536]),
)

train_dataloader = dict(batch_size=3)

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False,
        begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=False)
]

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=16)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05))

resume = True
