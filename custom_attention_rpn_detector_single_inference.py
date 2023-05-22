# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.
"""  # nowq

import os
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)
import cv2


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    # parser.add_argument('image', help='Image file')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    img = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/data/3963/3963/B0B81DF77L.jpg"
    config = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_fewshot_retrieval/3963/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.py"
    checkpoint = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/pipeline_fewshot_retrieval/3963/iter_1200.pth"
    support_images_dir = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/demo/demo_detection_images/support_images'
    device = 'cuda:0'
    score_thr = 0.1
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
    
    # test a single image
    result = inference_detector(model, img)
    print(result)
    # show the results
    show_result_pyplot(model, img, result, score_thr=score_thr)
    # img = cv2.imread(img)
    # print(img.shape)
    # for box in result[0]:
    #   if box[-1] >= score_thr:
    #     box = list(map(int, box))
    #     print(box)
    #     cropped = img[box[1]:box[3], box[0]:box[2]]
    #     cv2.imwrite("001.png", cropped)


if __name__ == '__main__':
    args = parse_args()
    main(args)
