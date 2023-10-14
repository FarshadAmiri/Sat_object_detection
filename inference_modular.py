from os import path, walk, makedirs, listdir
import math
import json
import csv
from PIL import Image
import PIL
import numpy as np
import torch
import supervision as sv
import cv2

from datasets import AirbusShipDetection
from imageutils import draw_bbox_torchvision
from tools import haversine_distance, shamsi_date_time
from slicing_inference import sahi_slicing_inference
from torchvision.transforms import v2 as tv2
from torchvision.ops import nms
from sahi import AutoDetectionModel


# ship_detection_sahi function takes the model path and image in PIL.Image.Image format and outputs 
# a dictionary with bboxes and respected scores after running Slicing Aid Hyper Inference (SAHI) on the image.
def ship_detection_single_image(image, model_path='models/best_model.pth', bbox_coord_wgs84=None, model_input_dim=768, sahi_confidence_threshold=0.9,
                   sahi_scale_down_factor='adaptive',sahi_overlap_ratio=0.2, nms_iou_threshold=0.1, output_scaled_down_image=True, device='adaptive'):
    
    # Set pytorch device.
    if device == 'adaptive':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    w, h = image.size
    transform = tv2.Compose([tv2.ToImageTensor(), tv2.ConvertImageDtype()])

    if sahi_scale_down_factor == 'adaptive':
        p = (w * h) // (model_input_dim * model_input_dim)
        if p > 10:
            sahi_scale_down_factor = int(round(math.sqrt(p / 20), 0))
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
                            output_scaled_down_image=True,
                            )
    bboxes = sahi_result["bboxes"]
    scores = sahi_result["scores"]
    scaled_down_image = sahi_result["scaled_down_image"] 
    scaled_down_image_size = sahi_result["scaled_down_image_size"]
    # image = transform(image) 
    
    # Perform Non-Max Suppression
    nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=nms_iou_threshold)
    bboxes = bboxes.numpy() 
    bboxes_nms = []
    bboxes_nms = np.array([bboxes[i] for i in nms_result])
    scores_nms = np.array([scores[i] for i in nms_result])

    # Output the result
    result = dict()
    result["n_obj"] = len(bboxes_nms)
    result["bboxes"] = bboxes_nms
    result["scores"] = scores_nms
    result["sahi_scaled_down_image"] = scaled_down_image

    # Calculating the longitude and latitude of each bbox's center as will as the detected ship length in meters (if bbox_coord_wgs84 is given):
    if bbox_coord_wgs84 != None:
        try:
            lon1, lat1, lon2, lat2 = bbox_coord_wgs84
        except:
            raise ValueError(f"""bbox_coord_wgs84 should be a python dictionary containing keys equal to the images name and
                            values equals to wgs84 coordinations in a list which is as follows:
                            [West Longitude , South Latitude , East Longitude , North Latitude]""")
        if (lon1 > lon2) or (lat1 > lat2):
            raise ValueError("""bbox_coord_wgs84 is supposed to be in the following format:
                                        [left, bottom, right, top]
                                        or in other words: 
                                        [min Longitude , min Latitude , max Longitude , max Latitude]
                                        or in other words: 
                                        [West Longitude , South Latitude , East Longitude , North Latitude]""")
        if any([(lon1 > 180), (lon2 > 180),
                (lon1 < -180), (lon2 < -180),
                (lat1 > 90), (lat2 > 90),
                (lat1 < -90), (lat2 < -90)]):
            raise ValueError("""Wrong coordinations! Latitude is between -90 and 90 and
                            Longitude is between -180 and 180. Also, the following format is required:
                            [left, bottom, right, top]
                            or in other words:
                            [min Longitude , min Latitude , max Longitude , max Latitude]
                            or in other words: 
                            [West Longitude , South Latitude , East Longitude , North Latitude]
                            """)
        
        w_resized, h_resized = scaled_down_image_size
        dist_h = haversine_distance(lat1, lon1, lat2, lon1)
        dist_w = haversine_distance(lat1, lon1, lat1, lon2)
        ships_coord = []
        ships_bbox_dimensions = []
        ships_length = []
        for bbox in bboxes_nms:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            ship_longitude = (((bbox_x1 + bbox_x2) * (lon2 - lon1)) / (2 * w_resized)) + lon1
            ship_longitude = round(ship_longitude, 12)
            ship_latitude = (((bbox_y1 + bbox_y2) * (lat2 - lat1)) / (2 * h_resized)) + lat1
            ship_latitude = round(ship_latitude, 12)
            ships_coord.append((ship_longitude, ship_latitude))

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

            result["ships_long-lat"] = ships_coord
            result["ships_length"] = ships_length
            result["ships_bbox_dimensions"] = ships_bbox_dimensions
    
    return result



# ship_detection_sahi function takes the model path and image in PIL.Image.Image format and outputs 
# a dictionary with bboxes and respected scores after running Slicing Aid Hyper Inference (SAHI) on the image.
def ship_detection(images_dir=None, images_objects=None, model_path='models/best_model.pth', bbox_coord_wgs84=None, model_input_dim=768, sahi_confidence_threshold=0.9,
                        sahi_scale_down_factor='adaptive', sahi_overlap_ratio=0.2, nms_iou_threshold=0.1, device='adaptive', output_dir=None,
                        save_annotated_images=True, output_original_image=True, output_annotated_image=False, annotations=["score", "length", "coord"]):
    
    # Check data validity (images_dir and images_objects)
    if (images_dir == None) and (images_objects == None):
        raise ValueError("""You should provide either images_dir or images_objects arguments.
                         you passed none of them into the function!""")
    elif (images_dir != None) and (images_objects != None):
        raise ValueError("""You should provide either images_dir or images_objects arguments.
                        you passed both of them into the function!""")
    
    # Preparing data in case images_dir is given.
    if images_dir != None:
        if path.exists(images_dir) == False:
            raise ValueError("""Please input a valid directory of images and make sure the path does not contain any space(' ') in it""")
        
        image_ext = ('.jpg','.jpeg', '.png', '.jp2', '.jfif', '.pjpeg', '.webp', '.tiff', '.tif')
        
        images_data = []
        for root, dirs, files in walk(images_dir):
            for filename in files:
                if filename.endswith(image_ext):
                    img = filename
                    if output_dir == None:
                        date_time = shamsi_date_time()
                        output_dir = path.join(images_dir, f"Predictions_{date_time}")
                        if not path.exists(output_dir):
                            makedirs(output_dir)
                    img_mask = path.join(output_dir, f"{path.splitext(filename)[0]}_pred{path.splitext(filename)[1]}")
                    img_size = Image.open(path.join(images_dir, filename)).size
                    images_data.append([img, img_mask, img_size])
            break
        del img, img_mask, img_size
    
    # Preparing data in case images_objects is given.
    elif images_objects != None:
        if type(images_objects) != list:
            if type(images_objects) not in [np.ndarray, PIL.Image.Image]:
                raise ValueError(f"""You passed an image which is not neither a np.ndarray nor PIL.Image.Image.
                                 images_objects must be a list of images objects or an image object in type of np.ndarray or PIL.Image.Image.""")
            elif type(images_objects) == np.ndarray:
                image = Image.fromarray(np.uint8(images_objects)).convert('RGB')
                images_objects = [image]
            else:
                images_objects = [images_objects]
        if (save_annotated_images== True) and (output_dir==None):
            raise ValueError("output_dir must be passed when save_annotated_images is True in images_objects mode!")
        elif (save_annotated_images== True) and (output_dir!=None):
            if not path.exists(output_dir):
                makedirs(output_dir)

        images_data = []  
        for idx, img in enumerate(images_objects):
            if type(img) not in [np.ndarray, PIL.Image.Image]:
                raise ValueError(f"""images_objects must be a list of images objects or an image object in type of np.ndarray or PIL.Image.Image.
                                 the element of indice {idx} is in type of {type(img)}!""")
            if type(img) == np.ndarray:
                img = Image.fromarray(np.uint8(img)).convert('RGB')
            if save_annotated_images:
                img_mask = path.join(output_dir, f"pred_{idx}.jpg")
            else:
                img_mask = 0
            img_size = img.size
            images_data.append([img, img_mask, img_size])

    # Set pytorch device
    if device == 'adaptive':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # transform = tv2.Compose([tv2.ToImageTensor(), tv2.ConvertImageDtype()])

    if sahi_scale_down_factor == 'adaptive':
        for img in images_data:
            w, h = img[2]
            p = (w * h) // (model_input_dim * model_input_dim)
            if p > 10:
                img_sahi_scale_down_factor = int(round(math.sqrt(p / 20), 0))
            else:
                img_sahi_scale_down_factor = 1
            img.append(img_sahi_scale_down_factor)

    model = torch.load(model_path, map_location = device)

    model = AutoDetectionModel.from_pretrained(
    model_type='torchvision',
    model=model,
    confidence_threshold=sahi_confidence_threshold,
    image_size=model_input_dim,
    device=device, # or "cuda:0"
    load_at_init=True,)

    result = dict()
    for img in images_data:
        if type(img) == str:
            img = path.join(images_dir, img[0])
        print(f"Processing {img[0]}")
        sahi_result = sahi_slicing_inference(image_or_dir=img, 
                                model=model, 
                                scale_down_factor=img[3], 
                                model_input_dim=model_input_dim, 
                                device=device, 
                                confidence_threshold=sahi_confidence_threshold, 
                                overlap_ratio=sahi_overlap_ratio,
                                output_scaled_down_image=True
                                    )
        bboxes = sahi_result["bboxes"]
        print(bboxes)
        scores = sahi_result["scores"]
        scaled_down_image_size = sahi_result["scaled_down_image_size"]
        sahi_scaled_down_image = sahi_result["scaled_down_image"]
        # image = transform(image) 
        
        # Perform Non-Max Suppression
        if bboxes.dim() == 1:
            result[img[0]] = dict()
            result[img[0]]["n_obj"] = 0
            result[img[0]]["bboxes"] = bboxes
            result[img[0]]["scores"] = scores
            result[img[0]]["sahi_scaled_down_image"] = sahi_scaled_down_image
            result[img[0]]["ships_coord"] = []
            result[img[0]]["ships_length"] = []
            result[img[0]]["ships_bbox_dimensions"] = []
            continue

        nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=nms_iou_threshold)
        bboxes = bboxes.numpy() 
        bboxes_nms = []
        bboxes_nms = np.array([bboxes[i] for i in nms_result])
        scores_nms = np.array([scores[i] for i in nms_result])

        # Output the result
        result[img[0]] = dict()
        result[img[0]]["n_obj"] = len(bboxes_nms)
        result[img[0]]["bboxes"] = bboxes_nms
        result[img[0]]["scores"] = scores_nms
        if output_original_image:
            result[img[0]]["sahi_scaled_down_image"] = sahi_scaled_down_image
        
        # Calculating the longitude and latitude of each bbox's center as will as the detected ship length in meters (if bbox_coord_wgs84 is given):
        if bbox_coord_wgs84 != None:
            if bbox_coord_wgs84.get(img[0]) != None:
                try:
                    lon1, lat1, lon2, lat2 = bbox_coord_wgs84[img[0]]
                except:
                    raise ValueError(f"""bbox_coord_wgs84 should be a python dictionary containing keys equal to the images name and
                                    values equals to wgs84 coordinations in a list which is as follows:
                                    [West Longitude , South Latitude , East Longitude , North Latitude]""")
                if (lon1 > lon2) or (lat1 > lat2):
                    raise ValueError("""bbox_coord_wgs84 is supposed to be in the following format:
                                                [left, bottom, right, top]
                                                or in other words: 
                                                [min Longitude , min Latitude , max Longitude , max Latitude]
                                                or in other words: 
                                                [West Longitude , South Latitude , East Longitude , North Latitude]""")
                if any([(lon1 > 180), (lon2 > 180),
                        (lon1 < -180), (lon2 < -180),
                        (lat1 > 90), (lat2 > 90),
                        (lat1 < -90), (lat2 < -90)]):
                    raise ValueError("""Wrong coordinations! Latitude is between -90 and 90 and
                                    Longitude is between -180 and 180. Also, the following format is required:
                                    [left, bottom, right, top]
                                    or in other words:
                                    [min Longitude , min Latitude , max Longitude , max Latitude]
                                    or in other words: 
                                    [West Longitude , South Latitude , East Longitude , North Latitude]
                                    """)
                
                w_resized, h_resized = scaled_down_image_size
                dist_h = haversine_distance(lat1, lon1, lat2, lon1)
                dist_w = haversine_distance(lat1, lon1, lat1, lon2)
                ships_coord = []
                ships_bbox_dimensions = []
                ships_length = []
                for bbox in bboxes_nms:
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
                    
                    ship_longitude = (((bbox_x1 + bbox_x2) * (lon2 - lon1)) / (2 * w_resized)) + lon1
                    ship_longitude = round(ship_longitude, 12)
                    ship_latitude = (((bbox_y1 + bbox_y2) * (lat2 - lat1)) / (2 * h_resized)) + lat1
                    ship_latitude = round(ship_latitude, 12)
                    ships_coord.append((ship_longitude, ship_latitude))

                    h_ship_bbox = ((bbox_y2 - bbox_y1) * dist_h) / h_resized
                    h_ship_bbox = round(h_ship_bbox, 1)
                    w_ship_bbox = ((bbox_x2 - bbox_x1) * dist_w) / w_resized
                    w_ship_bbox = round(w_ship_bbox, 1)
                    ships_bbox_dimensions.append((max(h_ship_bbox, w_ship_bbox), min(h_ship_bbox, w_ship_bbox)))

                    # Ship's length estimation:
                    if (h_ship_bbox / w_ship_bbox) >= 3 or (w_ship_bbox / h_ship_bbox) >= 3:
                        length = max(h_ship_bbox, w_ship_bbox)
                    else:
                        length = round(math.sqrt((h_ship_bbox ** 2) + (w_ship_bbox ** 2)), 1)
                    ships_length.append(length)

                    result[img[0]]["ships_long_lat"] = ships_coord
                    result[img[0]]["ships_length"] = ships_length
                    result[img[0]]["ships_bbox_dimensions"] = ships_bbox_dimensions
        
        # Drawing bbox and save image
        if save_annotated_images or output_annotated_image:
            if images_objects != None:
                output = img[1]
            else:
                output = path.join(images_dir, img[1])
            annotated_image = draw_bbox_torchvision(image=sahi_scaled_down_image, bboxes=bboxes_nms, scores=scores_nms,
                                  lengths=result[img[0]].get("ships_length"), ships_coords=result[img[0]].get("ships_long_lat"),
                                  annotations=annotations, save=save_annotated_images, image_save_name=output, output_annotated_image=output_annotated_image)
            if output_annotated_image:
                result[img[0]]["annotated_image"] = annotated_image
    del model
    return result


# remaining modules: 
# 2- csv or json result saving module [image hyperlink - time inteval - bbox coords - n_obj - lengths sorted dscending]
# 3- API module: takes bbox of an area - splitting it to allowable areas (sentinel-gub constraints) - download images of
# area in given time period [maybe splitting time into possible time intervals] and request images of all those from sentinel 
# api and run inference on all of them. result saved for each time interval and bbox coords [hyperlink the images in csv]
