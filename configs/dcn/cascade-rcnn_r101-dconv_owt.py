_base_ = [
    '../cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py',
    '../_base_/datasets/trash_detection.py',
        ]

train_dataloader = dict(batch_size=8)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[110.0754, 117.3973, 123.6507],
        std=[54.03457934, 53.36968771, 54.78390763],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
