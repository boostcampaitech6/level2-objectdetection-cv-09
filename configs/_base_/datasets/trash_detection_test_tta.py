dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}

custom_hooks = [dict(type='SubmissionHook')]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(
            type='nms',
            iou_threshold=0.5
        ),
        max_per_img=100
    )
)

img_scales = [(1024, 1024), (512, 512), (2048, 2048)]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(
                type='Resize',
                scale=s,
                keep_ratio=True) for s in img_scales],
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)],
            [dict(type='PackDetInputs')]
        ]
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_fold_4.json',
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_fold_4.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=tta_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_fold_4.json',
    metric='bbox',
    format_only=False)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=tta_pipeline))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'test.json',
    outfile_prefix='./work_dirs/trash/test')
