# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.
"""  # nowq

import os
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    # parser.add_argument('image', help='Image file')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    img = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/result/attention_rpn/door_20thr__attention_rpn/exclude/B0BQBZR9GB.jpg"
    config = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/door_attention-rpn_5shot-fine-tuning/door_attention-rpn_5shot-fine-tuning.py"
    checkpoint = "/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/work_dirs/door_attention-rpn_5shot-fine-tuning/iter_1200.pth"
    support_images_dir = '/home/tanluuuuuuu/Desktop/luunvt/image_retrieval/mmfewshot/demo/demo_detection_images/support_images'
    device = 'cuda:0'
    score_thr = 0
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


if __name__ == '__main__':
    args = parse_args()
    main(args)
