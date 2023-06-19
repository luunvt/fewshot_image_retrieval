from mmfewshot.detection.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm
import time
import cv2
import torch
import torchvision

# Specify the path to model config and checkpoint file
config_file = '/home/BI/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_v1/fine_tune_1/tfa_r101_fpn_voc-split1_5shot-fine-tuning_door_5000iter/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py'
checkpoint_file = '/home/BI/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_v1/fine_tune_1/tfa_r101_fpn_voc-split1_5shot-fine-tuning_door_5000iter/iter_5000.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_path = "/home/BI/luunvt/image_retrieval/data/image/GAR-514/GAR514"
sum_time = 0

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
  if len(result[-1]) > 0:
    output = result
    if len(result[-1]) > 1:
      boxes = torch.tensor(output[-1][:, :4]).squeeze()
      conf = torch.tensor(output[-1][:, 4:]).squeeze(1)
      if conf.shape[0] == 1:
        conf = torch.tensor(output[-1][:, 4:])
        nms_res = torchvision.ops.nms(boxes, conf, 0.1)
      color = (255, 0, 0)
      thickness = 0
      for box, score in zip(boxes[nms_res], conf[nms_res]):
          box_np = box.numpy().astype(int)
          score = score.numpy()
          img = cv2.rectangle(img, (box_np[0], box_np[1]), (box_np[2], box_np[3]), color, thickness)
          img = cv2.putText(img, str(score), (box_np[0], box_np[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
    else:
      box_np, score = result[-1][:, :4].astype(int), result[-1][:, 4:]
      color = (255, 0, 0)
      thickness = 0
      img = cv2.rectangle(img, (box_np[0][0], box_np[0][1]), (box_np[0][2], box_np[0][3]), color, thickness)
      img = cv2.putText(img, str(score), (box_np[0][0], box_np[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
    cv2.imwrite(f"/home/BI/luunvt/image_retrieval/mmfewshot/result/pipeline_v1/fine_tune_v2/tfa_r101_fpn_voc-split1_3shot-fine-tuning_papasan_5000iter_v2/include/{im_name}", img)
  else:
    cv2.imwrite(f"/home/BI/luunvt/image_retrieval/mmfewshot/result/pipeline_v1/fine_tune_v2/tfa_r101_fpn_voc-split1_3shot-fine-tuning_papasan_5000iter_v2/exclude/{im_name}", img)
print(f"Infer on {len(os.listdir(img_path))} got {sum_time}s")