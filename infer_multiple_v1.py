from mmfewshot.detection.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm
import time
import cv2
import torch
import torchvision
import numpy as np

THR = 0.5
THR_NMS = 0.3
# Specify the path to model config and checkpoint file
config_file = '/home/BI/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_v2/fine_tune_0/door_newbbox/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.py'
checkpoint_file = '/home/BI/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_v2/fine_tune_0/door_newbbox/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning/iter_4500.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_path = "/home/BI/luunvt/image_retrieval/data/img/door"
cat = "door"
# breakpoint()
num_iter = int(checkpoint_file.split("/")[-1].split(".")[0].split("_")[1])
folder_name = f"{cat}_new_bbox_{num_iter}iter_thr{str(int(THR*100))}_nms{str(int(THR_NMS*100))}"
save_path = "/home/BI/luunvt/image_retrieval/mmfewshot/result/pipeline_v2/fine_tune_0_v2/"

include_path = os.path.join(save_path, folder_name, "include")
if not os.path.exists(include_path):
  print(f"Make dir {include_path}")
  os.makedirs(include_path)

exclude_path = os.path.join(save_path, folder_name, "exclude")
if not os.path.exists(exclude_path):
  print(f"Make dir {exclude_path}")
  os.makedirs(exclude_path)

sum_time = 0

def nms(boxes):
  boxbox = torch.from_numpy(boxes[:, 0:4])
  score = torch.from_numpy(boxes[:, 4])
  result = torchvision.ops.nms(boxbox, score, THR_NMS)
  return result.numpy()

for im_name in tqdm(os.listdir(img_path)):
  img = os.path.join(img_path, im_name)
  st = time.time()
  try:
    result = inference_detector(model, img)
  except:
    print(im_name)
    continue
  sum_time += (time.time() - st)
  img = cv2.imread(img)
  boxes = np.array([box for box in result[-1] if box[4] >= THR])
  if len(boxes) > 0:
    box_idxs = nms(boxes)
    for box in boxes[box_idxs]:
      score = box[4]
      if score >= THR:
        box = box.astype(int)
        color = (255, 0, 0)
        thickness = 1
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
        img = cv2.putText(img, str(score), (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(include_path, im_name), img)
  else:
    cv2.imwrite(os.path.join(exclude_path, im_name), img)
print(f"Infer on {len(os.listdir(img_path))} got {sum_time}s")
