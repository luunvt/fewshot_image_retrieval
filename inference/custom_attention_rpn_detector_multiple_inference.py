# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.
"""  # nowq

import os
from tqdm import tqdm
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                    process_support_images)
import cv2
import numpy as np
import torch
import torchvision
import random


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    # parser.add_argument('image', help='Image file')
    args = parser.parse_args()
    return args

def nms(boxes, thr):
  boxbox = torch.from_numpy(boxes[:, 0:4])
  score = torch.from_numpy(boxes[:, 4])
  result = torchvision.ops.nms(boxbox, score, thr)
  return result.numpy()

def main():
    # build the model from a config file and a checkpoint file
    inference_image_path = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/data/img/door"
    save_output_image_path = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/result/attention_rpn/door_40thr_attention_rpn_v3"
    config = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/door_attention-rpn_5shot-fine-tuning/door_attention-rpn_5shot-fine-tuning.py"
    checkpoint = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/door_attention-rpn_5shot-fine-tuning/iter_1200.pth"
    support_images_dir = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/demo/demo_detection_images/support_images'
    device = 'cuda:0'
    score_thr = 0.40
    model = init_detector(config, checkpoint, device=device)

    # prepare support images, each demo image only contain one instance
    files = os.listdir(support_images_dir)
    support_images = [
        os.path.join(support_images_dir, file) for file in files
    ]

    classes = [file.split('.')[0] for file in files]
    support_labels = [[file.split('.')[0]] for file in files]
    process_support_images(
        model, support_images, support_labels, classes=classes)
    # breakpoint()
    for image_name in os.listdir(inference_image_path):
        image_path = os.path.join(inference_image_path, image_name)
        try:
            result = inference_detector(model, image_path)
            best_results = [np.expand_dims(sorted(result[0], key=lambda x: x[4])[-1], axis=0)]
            output_path = os.path.join(save_output_image_path, image_name)
            model.show_result(image_path, best_results, out_file=output_path, score_thr=score_thr)
        except:
            result = inference_detector(model, image_path)
            output_path = os.path.join(save_output_image_path, image_name)
            model.show_result(image_path, result, out_file=output_path, score_thr=score_thr)
        {
        # try:
        #     result = inference_detector(model, image_path)
        # except:
        #     print(image_path)
        #     continue
        # img = cv2.imread(image_path)
        # boxes = np.array([box for box in result[0] if box[4] >= score_thr])
        # if len(boxes) > 0:
        #     box_idxs = nms(boxes, 0.4)
        #     for box in boxes[box_idxs]:
        #         score = box[4]
        #         box = box.astype(int)
        #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #         thickness = 1
        #         img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
        #         img = cv2.putText(img, str(score), (box[0], box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
        #     cv2.imwrite(os.path.join(save_output_image_path, "include", image_name), img)
        # else:
        #     cv2.imwrite(os.path.join(save_output_image_path, "exclude", image_name), img)
        }


if __name__ == '__main__':
    {
        # args = parse_args()
        # main(args)
    }
    main()
