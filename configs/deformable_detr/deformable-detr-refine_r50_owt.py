_base_ = [
    'deformable-detr_r50_16xb2-50e_coco.py',
    '../_base_/datasets/trash_detection.py',
]
model = dict(
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[110.0754, 117.3973, 123.6507],
            std=[54.03457934, 53.36968771, 54.78390763],
            bgr_to_rgb=True,
            pad_size_divisor=32),
        with_box_refine=True)

train_dataloader = dict(batch_size=3)
