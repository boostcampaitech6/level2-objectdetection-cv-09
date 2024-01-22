_base_ = [
    './gfl_r50_fpn_ms-2x_coco.py',
    '../_base_/datasets/trash_detection.py'
]

train_dataloader = dict(batch_size=8)

model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[110.0754, 117.3973, 123.6507],
        std=[54.03457934, 53.36968771, 54.78390763],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))
