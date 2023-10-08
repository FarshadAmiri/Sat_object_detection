from os import path, walk, makedirs
import math
import json
import csv
from PIL import Image
import numpy as np
import torch
import supervision as sv
import cv2

from datasets import AirbusShipDetection
from imageutils import draw_image_with_boxes
from imageutils import resize_img_dir, resize_img
from slicing_inference import sahi_slicing_inference
from torchvision.transforms import v2 as Tv2
from torchvision.ops import nms


# Input folder with images to classify.
IMAGES_IN = r"inference_images"
IMAGES_IN = r"debug_images2"
IMAGES_IN = r"debug_images3"

# Folder with classified images (.png).
IMAGES_OUT = r"inference_predictions"

# CSV results file directory
RESULTS_DIR = r"results.csv"

# Must be model for binary classification.
MODEL_TO_USE = 'models/best_model.pth'

# Model input size of images.
IMG_DIM = 768

#IoU Threshold
iou_threshold = 0.08

# Confidence threshold
confidence_threshold = 0.85

#relative area minimum and maximum
area_min, area_max = 120, 100000000

# SAHI parameters
# How many times the size(area) of the photo should be bigger than IMG_DIM * IMG_DIM to activate SAHI inference (Default: 4)
SAHI_ACTIVATION_THRESHOLD = 4
# SAHI object detection score threshold
SAHI_CONFIDENCE_THRESHOLD = 0.88
# SAHI slicing boxes overlap with previous ones
SAHI_OVERLAP_RATIO = 0.25
# to resize very large images based on how large they are (adaptive)
SAHI_ADAPTIVE_RESIZING = True
# resize factor for very large pictures - No need to define if SAHI_ADAPTIVE_RESIZING is True
SAHI_SCALE_DOWN_FACTOR = 2

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
transform = Tv2.Compose([Tv2.ToImageTensor(), Tv2.ConvertImageDtype()])

extensionsToCheck = ['.png', '.jpg', '.jp2']

idx = 0
fn_images = []
for root, dirs, files in walk(IMAGES_IN):
    for fn_in in files:
        if any(ext in fn_in for ext in extensionsToCheck):

            # Create output filename.
            suffix = f'_{idx}_mask.png'

            if '.png' in fn_in:
                fn_out = fn_in.replace('.png', suffix)
            elif '.jpg' in fn_in:
                fn_out = fn_in.replace('.jpg', suffix)
            else:
                fn_out = fn_in.replace('.jp2', suffix)
            fn_images.append((path.join(root, fn_in), path.join(IMAGES_OUT, fn_out)))
            idx += 1

rows = []
for fn in fn_images:
    fn_in = fn[0]
    fn_out = fn[1]
    
    print(f'Processing {fn_in}')

    # Read the input image.
    image = Image.open(fn_in).convert("RGB")         # PIL: w,h    Numpy: h, w, _
    w, h = image.size

    # Init output image.
    image_out = np.empty((w, h, 3))

    # Init outputs: list of boxes and areas.
    bbox_list = []
    area_list = []


    if w * h > SAHI_ACTIVATION_THRESHOLD * IMG_DIM * IMG_DIM:
        if SAHI_ADAPTIVE_RESIZING == True:
            p = (w * h) // (IMG_DIM * IMG_DIM)
            if p > 10:
                SAHI_SCALE_DOWN_FACTOR = round(math.sqrt(p / 20), 1)
                print(f"SAHI_SCALE_DOWN_FACTOR set to: {SAHI_SCALE_DOWN_FACTOR}")
            else:
                SAHI_SCALE_DOWN_FACTOR = 1

        sahi_result = sahi_slicing_inference(image, 
                               model_dir=MODEL_TO_USE, 
                               scale_down_factor=SAHI_SCALE_DOWN_FACTOR, 
                               model_input_dim=IMG_DIM, 
                               device=device, 
                               confidence_threshold = SAHI_CONFIDENCE_THRESHOLD, 
                               overlap_ratio=0.2,
                               )
        bboxes = sahi_result["bboxes"]
        scores = sahi_result["scores"]
        image = sahi_result["resized_image"] 
        image = transform(image) 
    else:
        if  w != IMG_DIM or h != IMG_DIM:
            image = resize_img(image, IMG_DIM, IMG_DIM)
    
        image = transform(image)  #I_added

        # Apply the model to the image.
        x_tensor = image.to(device).unsqueeze(0)

        #x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        tgt_pred = model(x_tensor)
        tgt_pred = tgt_pred[0]

        # Get the boxes and apply the cropping offset + Applying Non-max suppression
        bboxes = tgt_pred['boxes'].detach().cpu()    #I_added
        # print(bboxes)
        scores= tgt_pred['scores'].detach().cpu()   #I_added
        # print(scores)
    nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=iou_threshold)
    bboxes = bboxes.numpy() 
    bboxes_nms = []
    bboxes_nms = np.array([bboxes[i] for i in nms_result])
    # print(bboxes_nms)
    # print(f"{fn_in} score:  {scores}")
    acceptable_scores_mask = np.array([i > confidence_threshold for i in scores])
    bboxes_nms = bboxes_nms[acceptable_scores_mask]
    scores = scores[acceptable_scores_mask]
    for bbox in bboxes_nms:

        # Threshold on area.
        # Maximum area: 360 * 50 m = 18000 m2 = 180 pixels.
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        # If object area is below threshold, add it to the object list.
        # if area >= 8 and area <= 180:   #I_commenetd
        if area >= area_min and area<= area_max:   #I_added

            bbox[0] += 0
            bbox[2] += 0
            bbox[1] += 0
            bbox[3] += 0
            bbox_list.append(bbox)

            area_list.append(area)

    # Convert pred_mask from `CHW` format to `HWC` format
    image = np.transpose(image, (1,2,0))

    # Save predicted mask.
    image_out = image

    n_objs = len(bbox_list)
    area_list_rounded= [ '%.2f' % elem for elem in area_list]
    # total_area = np.sum(area_list)
    # print(f'Detected {n_objs} objects in file {fn_in} with range between {time_from} to {time_to}.')   #I_commented
    print(f'Detected {n_objs} objects in file {fn_in}')

    row = [fn_in, 'time_from', 'time_to', n_objs, area_list_rounded]
    rows.append(row)

    # Save output mask.
    draw_image_with_boxes(fn_out, image_out, bbox_list)

# Save results to csv file.
with open(RESULTS_DIR, 'w', newline='') as csvfile:

    csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '|',
        quoting = csv.QUOTE_MINIMAL)

    csv_writer.writerow(['filename', 'from', 'to', 'n_objs', 'ships_areas'])

    for row in rows:
        csv_writer.writerow(row)