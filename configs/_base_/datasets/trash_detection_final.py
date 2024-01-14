dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}

img_scale = (512, 512)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale,
         center_ratio_range=(0.5, 1.0), prob=0.3),
    dict(
        type='CutOut',
        n_holes=7,
        cutout_shape=[
            (4, 4), (4, 8), (8, 4),
            (8, 8), (16, 8), (8, 16),
            (16, 16), (16, 32), (32, 16)]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            metainfo=metainfo,
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'train_fold_4.json',
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
