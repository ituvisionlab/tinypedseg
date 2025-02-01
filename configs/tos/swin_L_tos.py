_base_ = [
    '../_base_/models/upernet_swin.py'
]
#checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
model = dict(
    backbone=dict(
        window_size=12),
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(type='TverskyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=dict(type='TverskyLoss', loss_weight=1.0)),
        
    )

# dataset settings
dataset_type = 'TinyObjectSegmentationDataset'
data_root = './tinypedseg'  # Update with your data root
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
resize_size = (1024,2048)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=resize_size),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=resize_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/train/Images',
        ann_dir='dataset/train/Labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test/Images',
        ann_dir='dataset/test/Labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test/Images',
        ann_dir='dataset/test/Labels',
        pipeline=test_pipeline))
    
log_config = dict(
    interval=5, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = './work_dirs/swin'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
#optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=3600)
checkpoint_config = dict(by_epoch=False, interval=600)
evaluation = dict(interval=50, metric=['mIoU','mDice'], pre_eval=True, save_best = 'mDice')

seed = 0
gpu_ids = range(1)
device = None