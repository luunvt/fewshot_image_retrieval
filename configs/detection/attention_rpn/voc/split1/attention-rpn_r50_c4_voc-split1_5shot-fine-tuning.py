_base_ = [
    '../../../_base_/datasets/query_aware/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../attention-rpn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
num_support_ways = 2
num_support_shots = 4
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='Attention_RPN', setting='SPLIT1_5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
            min_bbox_area=0,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=400, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=400)
optimizer = dict(lr=0.001, momentum=0.9)
lr_config = dict(warmup=None, step=[1200])
log_config = dict(interval=10)
runner = dict(max_iters=1200)
# load_from = 'path of base training model'
load_from = \
    '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/base_model/attention-rpn_r50_c4_voc-split1_base-training_20211101_003606-58a8f413.pth'
model = dict(
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ],
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
