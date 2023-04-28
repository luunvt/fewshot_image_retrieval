from mmfewshot.detection.apis import init_detector, inference_detector
import mmcv
import numpy as np
import torch
import torchvision
import cv2

# Specify the path to model config and checkpoint file
config_file = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py'
checkpoint_file = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning/iter_800.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/data/img/focus_door/B093LV6YZB.jpg'  
# or img = mmcv.imread(img), which will only load it once
output = inference_detector(model, img)
result = output

# boxes = torch.tensor(output[-1][:, :4]).squeeze()
# conf = torch.tensor(output[-1][:, 4:]).squeeze()
# nms_res = torchvision.ops.nms(boxes, conf, 0.1)
# print(nms_res)
# print(boxes[nms_res])
# print(conf[nms_res])

# image = cv2.imread(img)  
# color = (255, 0, 0)
# thickness = 0
# for box, score in zip(boxes[nms_res], conf[nms_res]):
#     box_np = box.numpy().astype(int)
#     score = score.numpy()
#     print(box_np)
#     print(score)
#     image = cv2.rectangle(image, (box_np[0], box_np[1]), (box_np[2], box_np[3]), color, thickness)
#     image = cv2.putText(image, str(score), (box_np[0], box_np[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
# cv2.imwrite("work_dirs/pipeline_v1/fine_tune_2/tfa_r101_fpn_voc-split1_5shot-fine-tuning_5000iter_v2/result_nms.jpg", image)

model.show_result(img, result, out_file='result.jpg')