#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply model for ship detection on Sentinel-2 images."""
from os import path, walk, makedirs
import json
import csv
from PIL import Image
import numpy as np
import torch
import supervision as sv

from datasets import AirbusShipDetection
from imageutils import draw_image_with_boxes
# from transforms import ToTensor
from torchvision.transforms import v2 as Tv2
from torchvision.ops import nms

# Input folder with images to classify.
# IMAGES_IN = r"D:\NLP 1\s2-ship-detection\inference_images"
IMAGES_IN = r"D:\NLP 1\Sat_object_detection\inference_images"

# Folder with classified images (.png).
IMAGES_OUT = r"D:\NLP 1\Sat_object_detection\inference_predictions"

# CSV results file directory
RESULTS_DIR = r"D:\NLP 1\Sat_object_detection\results.csv"

# Must be model for binary classification.
MODEL_TO_USE = 'best_model.pth'
# MODEL_TO_USE = 'best_model_epoch2.pth'

# Model input size of images.
IMG_DIM = 768

#IoU Threshold
iou_threshold = 0.08

#relative area minimum and maximum
area_min, area_max = 120, 1000000

# Set pytorch device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not path.exists(IMAGES_OUT):
    makedirs(IMAGES_OUT)

# Load best saved model checkpoint from previous commit (if present).
model = torch.load(MODEL_TO_USE, map_location = device)
print('Model loaded.')

model.eval()
print('Model evaluated.')

# Transformation of images.
# transform = ToTensor()  #I_commented
transform = Tv2.Compose([Tv2.ToImageTensor(), Tv2.ConvertImageDtype()])

extensionsToCheck = ['.png', '.jpg', '.jp2']

idx = 0
fn_images = []
for r, d, f in walk(IMAGES_IN):

    for fn_in in f:

        if any(ext in fn_in for ext in extensionsToCheck):

            # Create output filename.
            suffix = f'_{idx}_mask.png'

            if '.png' in fn_in:
                fn_out = fn_in.replace('.png', suffix)
            elif '.jpg' in fn_in:
                fn_out = fn_in.replace('.jpg', suffix)
            else:
                fn_out = fn_in.replace('.jp2', suffix)

            fn_images.append((path.join(r, 'request.json'), path.join(r, fn_in), path.join(IMAGES_OUT, fn_out)))

            idx += 1

rows = []

for fn in fn_images:

    fn_in = fn[1]
    fn_out = fn[2]

    print(f'Processing {fn_in}...')

    # Read the request.
    # with open(fn[0], 'r', encoding='utf-8') as file:
    #     request = json.load(file)
    # data = request['payload']['input']['data'][0]
    # time_range = data['dataFilter']['timeRange']
    # time_from = time_range['from']
    # time_to = time_range['to']

    # Read the input image.
    image_in = Image.open(fn_in).convert("RGB")
    n_rows, n_cols = image_in.size

    # Init output image.
    image_out = np.empty((n_rows, n_cols, 3))

    # Init outputs: list of boxes and areas.
    box_list = []
    area_list = []

    # Predict image samples.
    for ii in range(0, n_rows, IMG_DIM):
        for jj in range(0, n_cols, IMG_DIM):

            if (ii + IMG_DIM) > n_rows or (jj + IMG_DIM) > n_cols:
                #print(f'{ii},{jj} indices run out of bounds, skipping.')
                continue

            cropping_box = (jj, ii, jj + IMG_DIM, ii + IMG_DIM)
            image = image_in.crop(cropping_box)
            # ox1 = transform(image)
            # print('type: ', type(ox1))
            # print('len: ', len(ox1))
            # print('ox1.shape: ', ox1.shape)
            # print('ox1[0].shape: ', ox1[0].shape)
            # image, _ = transform(image)  #I_commented
            image = transform(image)  #I_added

            # Apply the model to the image.
            x_tensor = image.to(device).unsqueeze(0)
            #x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            tgt_pred = model(x_tensor)
            tgt_pred = tgt_pred[0]

            # Get the boxes and apply the cropping offset + Applying Non-max suppression
            boxes = tgt_pred['boxes'].detach().cpu()    #I_added
            scores= tgt_pred['scores'].detach().cpu()   #I_added
            nms_result = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
            boxes = boxes.numpy() 
            boxes_nms = []
            boxes_nms = [boxes[i] for i in nms_result]
            for box in boxes_nms:

                # Threshold on area.
                # Maximum area: 360 * 50 m = 18000 m2 = 180 pixels.
                area = (box[3] - box[1]) * (box[2] - box[0])

                # If object area is below threshold, add it to the object list.
                # if area >= 8 and area <= 180:   #I_commenetd
                if area >= area_min and area<= area_max:   #I_added

                    box[0] += ii
                    box[2] += ii
                    box[1] += jj
                    box[3] += jj
                    box_list.append(box)

                    area_list.append(area)

            # Convert pred_mask from `CHW` format to `HWC` format
            image = np.transpose(image, (1,2,0))

            # Save predicted mask.
            image_out[ii : ii + IMG_DIM, jj : jj + IMG_DIM, :] = image

    n_objs = len(box_list)
    area_list_rounded= [ '%.2f' % elem for elem in area_list]
    # total_area = np.sum(area_list)
    # print(f'Detected {n_objs} objects in file {fn_in} with range between {time_from} to {time_to}.')   #I_commented
    print(f'Detected {n_objs} objects in file {fn_in} with range between time_from to time_to.')

    row = [fn_in, 'time_from', 'time_to', n_objs, area_list_rounded]
    rows.append(row)

    # Save output mask.
    draw_image_with_boxes(fn_out, image_out, box_list)

# Save results to csv file.
with open(RESULTS_DIR, 'w', newline='') as csvfile:

    csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '|',
        quoting = csv.QUOTE_MINIMAL)

    csv_writer.writerow(['filename', 'from', 'to', 'n_objs', 'ships_areas'])

    for row in rows:
        csv_writer.writerow(row)
