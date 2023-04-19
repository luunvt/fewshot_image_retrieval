import cv2
import numpy as np
import os

## Convert to gray, and threshold
def draw_bbox(input_path, output_path):
  img = cv2.imread(input_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

  ## Morph-op to remove noise
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
  morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

  ## Find the max-area contour
  cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
  # cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
  breakpoint()
  ## Crop and save 
  x1 = []
  y1 = []
  x2 = []
  y2 = []
  for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    x1.append(x)
    y1.append(y)
    x2.append(x+w)
    y2.append(y+h)
  x1 = min(x1)
  y1 = min(y1)
  x2 = max(x2)
  y2 = max(y2)

  color = (0, 0, 255)
  thickness = 2
  img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
  cv2.imwrite(output_path, img)

# path = "/home/BI/luunvt/image_retrieval/data/all_focus"
# out_path = "/home/BI/luunvt/image_retrieval/data/all_focus_bbox"
# for folder_name in os.listdir(path):
#   for img_name in os.listdir(os.path.join(path, folder_name)):
#     input_path = os.path.join(path, folder_name, img_name)
#     output_path = os.path.join(out_path, img_name)
#     draw_bbox(input_path, output_path)
draw_bbox("/home/BI/luunvt/image_retrieval/data/photo_2023-04-11_08-17-05 (3).jpg", "002.png")