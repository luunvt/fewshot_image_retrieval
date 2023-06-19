from mmfewshot.detection.apis import init_detector, inference_detector
import mmcv
import numpy as np
import torch
import torchvision
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img', help='image path')
    parser.add_argument('--output', help='output path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # Specify the path to model config and checkpoint file
    config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    img = args.img  

    output = inference_detector(model, img)
    result = output

    model.show_result(img, result, out_file=args.output)