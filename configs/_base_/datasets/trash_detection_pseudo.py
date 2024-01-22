dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}

albu_train_transforms = [
    dict(
        type='CLAHE',
        p=0.3),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.3),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        p=0.2),
    dict(
        type='Sharpen',
        alpha=[0.1, 0.3],
        lightness=[0.5, 1.0],
        p=0.2),
    dict(
        type='Emboss',
        alpha=0.2,
        p=0.2),
    dict(
        type='JpegCompression',
        quality_lower=85,
        quality_upper=95,
        p=0.2)
]

img_scale = (512, 512)

train_pipeline = [
    dict(
        type='Mosaic', img_scale=img_scale,
        center_ratio_range=(0.5, 1.0), prob=0.3),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            metainfo=metainfo,
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'pseudo_label.json',
            data_prefix=dict(img=data_root),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
        ),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file='val_fold_4.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_fold_4.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
