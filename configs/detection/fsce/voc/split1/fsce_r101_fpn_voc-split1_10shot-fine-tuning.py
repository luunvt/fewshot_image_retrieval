_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../fsce_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT1_10SHOT')],
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=7500,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=7500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[8000, 13000])
runner = dict(max_iters=15000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
work_dir = './work_dirs/fsce_r101_fpn_voc-split1_10shot-fine-tuning'
load_from = ('work_dirs/fsce_r101_fpn_voc-split1_base-training/'
             'base_model_random_init_bbox_head.pth')
