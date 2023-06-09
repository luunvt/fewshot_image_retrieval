
# Fewshot detect to retrieve

Few shot learning aims at generalizing to new tasks based on a limited number of samples using prior knowledge. The prior knowledge usually refers to a large scale training set that has many classes and samples, while the samples in new tasks are never seen in the training set. In few shot detection, a detector needs to detect the new categories based on a few instances.

In summary, few shot learning focus on two aspects:
* pre-train with large scale dataset.
* learn on a few labeled samples.




## Installation
Requires:
cuda - 11.8

```bash
  conda create -n mmfewshot_test python=3.8.10 -y
  conda activate mmfewshot_test
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

  pip install openmim
  mim install "mmcv-full==1.6.0"

  mim install mmcls
  mim install "mmdet==2.25.0"

  git clone git@github.com:luunvt/fewshot_image_retrieval.git
  cd fewshot_image_retrieval
  pip install -r requirements/build.txt
  pip install -v -e .
```
    
## Training
1. Download pretrained base model
```bash
wget https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training_20211101_003606-58a8f413.pth
```
2. Edit categories in `./mmfewshot/detection/datasets/voc.py`
3. Add image to path
4. Train
```bash
CUDA_VISIBLE_DEVICES=0 python tools/detection/train.py configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.py --no-validate
```

## Inference
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer_single.py --config <config path> --checkpoint <checkpoint path> --img <image path> --output <output path>
```