_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../fsce_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT1_2SHOT')],
        num_novel_shots=2,
        num_base_shots=2,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=3500,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=3500)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=20, step=[
        5000,
    ])
runner = dict(max_iters=7000)
model = dict(frozen_parameters=[
    'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs.0'
])
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
work_dir = './work_dirs/fsce_r101_fpn_voc-split1_2shot-fine-tuning'
load_from = ('work_dirs/fsce_r101_fpn_voc-split1_base-training/'
             'base_model_random_init_bbox_head.pth')
