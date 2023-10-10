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
from tools import distance
from slicing_inference import sahi_slicing_inference
from torchvision.transforms import v2 as Tv2
from torchvision.ops import nms


# ship_detection_standard function takes the model and image in PIL.Image.Image format and outputs 
# a dictionary with bboxes and respected scores in retrun.
# def ship_detection_standard(image, model, bbox_coord_wgs84=None, model_input_dim=768, confidence_threshold=0.85, nms_iou_threshold=0.1, device='adaptive'):
       
#     # Set pytorch device.
#     if device == 'adaptive':
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device(device)

#     w, h = image.size
#     transform = Tv2.Compose([Tv2.ToImageTensor(), Tv2.ConvertImageDtype()])

#     if  w != model_input_dim or h != model_input_dim:
#         image = resize_img(image, model_input_dim, model_input_dim)
    
#     image = transform(image)

#     # Apply the model to the image.
#     x_tensor = image.to(device).unsqueeze(0)

#     #x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
#     prediction = model(x_tensor)
#     prediction = prediction[0]

#     # Get the boxes and apply the cropping offset + Applying Non-max suppression
#     bboxes = prediction['boxes'].detach()
#     scores= prediction['scores'].detach()
    
#     try:
#         bboxes = bboxes.cpu()
#         scores = scores.cpu()
#     except:
#         pass

#     # Perform Non-Max Suppression
#     nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=nms_iou_threshold)
#     bboxes = bboxes.numpy() 
#     bboxes_nms = []
#     bboxes_nms = np.array([bboxes[i] for i in nms_result])
#     scores_nms = np.array([scores[i] for i in nms_result])

#     # remove bboxes with probability less than confidence_threshold
#     acceptable_scores_mask = np.array([i > confidence_threshold for i in scores_nms])
#     bboxes_nms = bboxes_nms[acceptable_scores_mask]
#     scores_nms = scores_nms[acceptable_scores_mask]

    # # Calculating the longitude and latitude of each bbox's center as will as the detected ship length in meters (if bbox_coord_wgs84 is given):
    # if bbox_coord_wgs84 != None:
    #     if (bbox_coord_wgs84[0] > bbox_coord_wgs84[2]) or (bbox_coord_wgs84[1] > bbox_coord_wgs84[3]):
    #         raise ValueError("""bbox_coord_wgs84 is supposed to be in the following format:
    #                                      [left, bottom, right, top]
    #                                      or in other words: 
    #                                      [min Longitude , min Latitude , max Longitude , max Latitude]
    #                                      or in other words: 
    #                                      [West Longitude , South Latitude , East Longitude , North Latitude]""")
    #     if any([(bbox_coord_wgs84[0] > 180), (bbox_coord_wgs84[2] > 180),
    #             (bbox_coord_wgs84[0] < -180), (bbox_coord_wgs84[2] < -180)],
    #             (bbox_coord_wgs84[1] > 90), (bbox_coord_wgs84[3] > 90),
    #             (bbox_coord_wgs84[1] < -90), (bbox_coord_wgs84[3] < -90)):
    #         raise ValueError("""Wrong coordinations! Latitude is between -90 and 90 and
    #                          Longitude is between -180 and 180. Also, the following format is required:
    #                          [left, bottom, right, top]
    #                          or in other words:
    #                          [min Longitude , min Latitude , max Longitude , max Latitude]
    #                          or in other words: 
    #                          [West Longitude , South Latitude , East Longitude , North Latitude]
    #                          """)
    #     # bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes_nms
    #     # cx =



#     result = dict()
#     result["n_obj"] = len(bboxes_nms)
#     result["bboxes"] = bboxes_nms
    # result["scores"] = scores_nms
    
    # return result


# ship_detection_sahi function takes the model path and image in PIL.Image.Image format and outputs 
# a dictionary with bboxes and respected scores after running Slicing Aid Hyper Inference (SAHI) on the image.
def ship_detection_sahi(image, model_path='models/best_model.pth', bbox_coord_wgs84=None, model_input_dim=768, sahi_confidence_threshold=0.9, sahi_scale_down_factor='adaptive',
                        sahi_overlap_ratio=0.2, nms_iou_threshold=0.1, output_scaled_down_image=True, device='adaptive'):
    
    # Set pytorch device.
    if device == 'adaptive':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    w, h = image.size
    transform = Tv2.Compose([Tv2.ToImageTensor(), Tv2.ConvertImageDtype()])

    if sahi_scale_down_factor == 'adaptive':
        p = (w * h) // (model_input_dim * model_input_dim)
        if p > 10:
            sahi_scale_down_factor = round(math.sqrt(p / 20), 1)
            print(f"SAHI_SCALE_DOWN_FACTOR set to: {int(sahi_scale_down_factor)}")
        else:
            sahi_scale_down_factor = 1

    sahi_result = sahi_slicing_inference(image_or_dir=image, 
                            model_dir=model_path, 
                            scale_down_factor=sahi_scale_down_factor, 
                            model_input_dim=model_input_dim, 
                            device=device, 
                            confidence_threshold=sahi_confidence_threshold, 
                            overlap_ratio=sahi_overlap_ratio,
                            output_scaled_down_image=output_scaled_down_image
                            )
    bboxes = sahi_result["bboxes"]
    scores = sahi_result["scores"]
    image = sahi_result["scaled_down_image"] 
    res_object = sahi_result["res_object"]    #remove before release
    # image = transform(image) 
    
    # Perform Non-Max Suppression
    nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=nms_iou_threshold)
    bboxes = bboxes.numpy() 
    bboxes_nms = []
    bboxes_nms = np.array([bboxes[i] for i in nms_result])
    scores_nms = np.array([scores[i] for i in nms_result])

    # Output the result
    result = dict()
    result["res_object"] = res_object
    result["n_obj"] = len(bboxes_nms)
    result["bboxes"] = bboxes_nms
    result["scores"] = scores_nms
    result["sahi_scaled_down_image"] = image

    # Calculating the longitude and latitude of each bbox's center as will as the detected ship length in meters (if bbox_coord_wgs84 is given):
    if bbox_coord_wgs84 != None:
        lg1, lt1, lg2, lt2 = bbox_coord_wgs84
        if (lg1 > lg2) or (lt1 > lt2):
            raise ValueError("""bbox_coord_wgs84 is supposed to be in the following format:
                                         [left, bottom, right, top]
                                         or in other words: 
                                         [min Longitude , min Latitude , max Longitude , max Latitude]
                                         or in other words: 
                                         [West Longitude , South Latitude , East Longitude , North Latitude]""")
        if any([(lg1 > 180), (lg2 > 180),
                (lg1 < -180), (lg2 < -180),
                (lt1 > 90), (lt2 > 90),
                (lt1 < -90), (lt2 < -90)]):
            raise ValueError("""Wrong coordinations! Latitude is between -90 and 90 and
                             Longitude is between -180 and 180. Also, the following format is required:
                             [left, bottom, right, top]
                             or in other words:
                             [min Longitude , min Latitude , max Longitude , max Latitude]
                             or in other words: 
                             [West Longitude , South Latitude , East Longitude , North Latitude]
                             """)
        
        w_resized, h_resized = image.size
        dist_h = distance(lt1, lt2, lg1, lg1)
        dist_w = distance(lt1, lt1, lg1, lg2)
        ships_coord = []
        ships_bbox_dimensions = []
        ships_length = []
        for bbox in bboxes_nms:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            cx = (((bbox_x1 + bbox_x2) * (lg2 - lg1)) / (2 * w_resized)) + lg1
            cx = round(cx, 12)
            cy = (((bbox_y1 + bbox_y2) * (lt2 - lt1)) / (2 * h_resized)) + lt1
            cy = round(cy, 12)
            ships_coord.append((cx, cy))

            h_ship_bbox = ((bbox_y2 - bbox_y1) * dist_h) / h_resized
            h_ship_bbox = round(h_ship_bbox, 1)
            w_ship_bbox = ((bbox_x2 - bbox_x1) * dist_w) / w_resized
            w_ship_bbox = round(w_ship_bbox, 1)
            ships_bbox_dimensions.append((max(h_ship_bbox, w_ship_bbox), min(h_ship_bbox, w_ship_bbox)))

            # Ship's length estimation:
            if (h_ship_bbox / w_ship_bbox) >= 2.5 or (w_ship_bbox / h_ship_bbox) >= 2.5:
                length = max(h_ship_bbox, w_ship_bbox)
            else:
                length = round(math.sqrt((h_ship_bbox ** 2) + (w_ship_bbox ** 2)), 1)
            ships_length.append(length)

            result["ships_coord"] = ships_coord
            result["ships_length"] = ships_length
            result["ships_bbox_dimensions"] = ships_bbox_dimensions

    
    return result


# if input is single image(PIL.Image.Image or np.ndarray) -> output n_obj , bbox , score , (lt_c , lg_c), length and
# save annotated image where user specified.

# if input is a folder -> output n_obj , bbox , score , (lt , lg), length and
# and save annotated image and results.csv where user specified.
# results.csv [image]
def ship_detection(image_or_dir, model_path='models/best_model.pth', sahi_mode='adaptive', bbox_coord_wgs84=None, model_input_dim=768, 
                   sahi_confidence_threshold=0.9, sahi_scale_down_factor='adaptive', sahi_overlap_ratio=0.2,
                   nms_iou_threshold=0.1, output_scaled_down_image=True, device='adaptive'):
    bulk_mode = False
    if type(image_or_dir) == str:    # if an image's path or a directory containing images is passed into the function
        image_ext = ['.jpg','.jpeg', '.png', '.jp2', '.jfif', '.pjpeg', '.webp', '.tiff', '.tif']
        if any(ext in image_path[-6:] for ext in image_ext):
            image = Image.open(image_or_dir)
        else:
            images_dir = image_or_dir
            bulk_mode = True
    elif type(image_or_dir) == np.ndarray:
        image = Image.fromarray(np.uint8(image_or_dir)).convert('RGB')
    elif type(image_or_dir) != Image.Image:
        raise TypeError("image_or_dir should be whether a np.ndarray or PIL.Image.Image or a directory string")
    else:
        image = image_or_dir    # in case the image passed is a PIL.Image.Image object.
    
    if bulk_mode == False:
        if sahi_mode:
            result = ship_detection_sahi(image, model_path=model_path, coord=coord, model_input_dim=model_input_dim,
                                sahi_confidence_threshold=sahi_confidence_threshold, sahi_scale_down_factor=sahi_scale_down_factor,
                                sahi_overlap_ratio=sahi_overlap_ratio, nms_iou_threshold=nms_iou_threshold, 
                                output_scaled_down_image=output_scaled_down_image, device=device)
                                


    


    result = dict() 
    return result










def ship_detection_bulk():
    # load_image
    if image == str:
        image = Image.open(image).convert("RGB")
    elif type(image) == np.ndarray:
        image = Image.fromarray(np.uint8(image)).convert('RGB')
    
    if type(image) != Image.Image:
        raise TypeError("Input image should be in type of np.ndarray or PIL.Image.Image")
    
    w, h = image.size


    # Init output image.
    image_out = np.empty((w, h, 3)) 

    # Init outputs: list of boxes and areas.
    bbox_list = []
    area_list = []
    






def detect_ships(image, image_out=True, image_save_dir=None, model_path='models/best_model.pth', model_input_size=768, device=None, confidence_threshold=0.85
                 , area_min=120, area_max=1e10, nms_iou_threshold=0.1, sahi_activation_threshold=4, sahi_confidence_threshold=0.88
                 , sahi_overlap_ratio=0.25, sahi_adaptive_resizing=True, sahi_scale_down_factor=2):
    
    # Set pytorch device.
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if not path.exists(image_save_dir):
        makedirs(image_save_dir)

    # Load best saved model checkpoint from previous commit (if present).
    model = torch.load(MODEL_TO_USE, map_location = device)
    print('Model loaded.')

    model.eval()
    print('Model evaluated.')

    # Transformation of images.
    transform = Tv2.Compose([Tv2.ToImageTensor(), Tv2.ConvertImageDtype()])

    # load_image
    if image == str:
        image = Image.open(image).convert("RGB")
    elif type(image) == np.ndarray:
        h, w, _ = image.shape
        image = Image.fromarray(np.uint8(image)).convert('RGB')
    

    print(f'Processing image')

    # Read the input image.
           # PIL: w,h    Numpy: h, w, _
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
    
        image = transform(image)  

        # Apply the model to the image.
        x_tensor = image.to(device).unsqueeze(0)

        #x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        tgt_pred = model(x_tensor)
        tgt_pred = tgt_pred[0]

        # Get the boxes and apply the cropping offset + Applying Non-max suppression
        bboxes = tgt_pred['boxes'].detach().cpu()    
        scores= tgt_pred['scores'].detach().cpu()   
        # print(bboxes)
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
        if area >= area_min and area<= area_max:   

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

    result = dict
    return result








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




