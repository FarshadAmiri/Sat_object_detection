from os import path, walk, makedirs
import math
from PIL import Image
import PIL
import numpy as np
import torch

from imageutils import draw_bbox_torchvision
from tools import haversine_distance, shamsi_date_time, is_image, verify_coords, calculate_scale_down_factor
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
def ship_detection(images, model_or_model_path='models/best_model.pth', bbox_coord_wgs84=None, model_input_dim=768, confidence_threshold=0.9,
                   scale_down_factor='adaptive', adaptive_scale_down_parameters = {'a': 0.2, 'b': 0.75, 'threshold': 1.5}, sahi_overlap_ratio=0.2,
                   nms_iou_threshold=0.1, device='adaptive', output_dir=None, output_name="prediction",save_annotated_image=False,
                   output_original_image=False, output_annotated_image=True, annotations=["score", "length", "coord"],
                   annotation_font=r"calibri.ttf",annotation_font_size=14, annotation_bbox_width=2):
    
    # Check data validity (images_dir and images_objects)
    if type(images) == dict:
        inference_mode = "images_dict"
        images_dict = images
    elif is_image(images):
        inference_mode = "single_image"
        single_image = images
    elif type(images) == str:
        inference_mode = "directory"
        images_dir = images
    else:
        raise ValueError("""'images' must be an images directory or a single image(np.ndarray or PIL images) or a dictionary containing images with the following format:\n{'image_name1': {'image': PIL.Image.Image, 'coord':(53.78436, 24.16872)}, 'image_name2': {'image': np.ndarray}}""")
    del images
    
    # Preparing data in case images_dir is given.
    images_data = []
    if inference_mode == "directory":
        if path.exists(images_dir) == False:
            raise ValueError("""Please input a valid directory of images and make sure the path does not contain any space(' ') in it""")
        
        image_ext = ('.jpg','.jpeg', '.png', '.jp2', '.jfif', '.pjpeg', '.webp', '.tiff', '.tif')
        
        for root, dirs, files in walk(images_dir):
            for filename in files:
                if filename.endswith(image_ext):
                    if save_annotated_image:
                        img_name = filename
                        if output_dir == None:
                            date_time = shamsi_date_time()
                            output_dir = path.join(images_dir, f"Predictions_{date_time}")
                            if not path.exists(output_dir):
                                makedirs(output_dir)
                        img_mask = path.join(output_dir, f"{path.splitext(filename)[0]}_pred{path.splitext(filename)[1]}")
                    else:
                        img_mask = 0
                    img_size = Image.open(path.join(images_dir, filename)).size
                    images_data.append([img_name, img_mask, img_size])   #image_name= ok   img_mask = 0 or full_out_dir
            break
    
    # Preparing data in case images_dict is given.
    elif inference_mode == "images_dict":
        images_names = list(images_dict.keys())
        if (save_annotated_image== True) and (output_dir==None):
            raise ValueError("output_dir must be passed when save_annotated_images is True on images dictionary mode!")
        elif (save_annotated_image== True) and (output_dir!=None):
            if not path.exists(output_dir):
                makedirs(output_dir)

        for idx, img_name in enumerate(images_names):
            img = images_dict[img_name]["image"]
            if is_image(img) == False:
                raise ValueError(f"""images dictionary must contain images objects in type of np.ndarray or PIL images.\nthe object {img_name} is in type of {type(img)}!""")
            if type(img) == np.ndarray:
                img = Image.fromarray(np.uint8(img)).convert('RGB')
            if save_annotated_image:
                img_mask = path.join(output_dir, f"{img_name}_pred.jpg")
            else:
                img_mask = 0
            img_size = img.size
            images_data.append([img_name, img_mask, img_size])    #image_name= ok   img_mask = 0 or full_out_dir

    # Preparing data in case single_image is given.
    elif inference_mode == "single_image":
        if is_image(single_image) == False:
            raise ValueError(f"""Wrong input! image must be in np.ndarray or PIL images type.""")
        if type(single_image) == np.ndarray:
            single_image = Image.fromarray(np.uint8(single_image)).convert('RGB')
        if (save_annotated_image== True) and (output_dir==None):
            raise ValueError("output_dir must be passed when save_annotated_images is True in single image mode!")
        elif (save_annotated_image== True) and (output_dir!=None):
            if not path.exists(output_dir):
                makedirs(output_dir)
        image_ext = ('.jpg','.jpeg', '.png', '.jp2', '.jfif', '.pjpeg', '.webp', '.tiff', '.tif')
        if save_annotated_image:
            if output_dir.endswith(image_ext) == False:
                if output_name.endswith(image_ext) == False:
                    output_name = output_name + '.jpg'
                img_mask = path.join(output_dir, output_name)
            else:
                img_mask = output_name
        else:
            img_mask = 0
        img_size = single_image.size
        img_name = output_name
        
        images_data.append([img_name, img_mask, img_size])   #image_name= ok   img_mask = 0 or full_out_dir

    del img_name, img_mask, img_size

    # Set pytorch device
    if device == 'adaptive':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # transform = tv2.Compose([tv2.ToImageTensor(), tv2.ConvertImageDtype()])

    for img in images_data:
        if scale_down_factor == 'adaptive':
            a = adaptive_scale_down_parameters['a']
            b = adaptive_scale_down_parameters['b']
            threshold = adaptive_scale_down_parameters['threshold']
            area = img[2][0] * img[2][1]
            
            scale_down_rate = calculate_scale_down_factor(area, model_input_dim ,a ,b ,threshold)
        else:
            try:
                scale_down_rate = float(scale_down_factor)
            except:
                raise ValueError('scale_down_factor should be either set to "adaptive" or set to a number (int or float)')
        # print(f"scale_down_rate: {scale_down_rate}")
        img.append(scale_down_rate)

    if type(model_or_model_path) == str:
        model = torch.load(model_or_model_path, map_location = device)

        model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=confidence_threshold,
        image_size=model_input_dim,
        device=device, # or "cuda:0"
        load_at_init=True,)

    result = dict()
    for img in images_data:
        if inference_mode == "directory":
            img_dir_or_object = path.join(images_dir, img[0])
        elif inference_mode == "images_dict":
            img_dir_or_object = images_dict[img_name]["image"]
        elif inference_mode == "single_image":
            img_dir_or_object = single_image
        print(f"Processing {img[0]}")
        sahi_result = sahi_slicing_inference(image_or_dir=img_dir_or_object, 
                                model=model, 
                                scale_down_factor=img[3], 
                                model_input_dim=model_input_dim, 
                                device=device, 
                                confidence_threshold=confidence_threshold, 
                                overlap_ratio=sahi_overlap_ratio,
                                output_scaled_down_image=any([save_annotated_image, output_original_image, output_annotated_image])
                                    )
        bboxes = sahi_result["bboxes"]
        scores = sahi_result["scores"]
        scaled_down_image_size = sahi_result["scaled_down_image_size"]
        sahi_scaled_down_image = sahi_result.get("scaled_down_image")
        # image = transform(image) 
        
        # Perform Non-Max Suppression
        if bboxes.dim() == 1:
            result[img[0]] = dict()
            if inference_mode == "single_image":
                result["n_obj"] = 0
                result["bboxes"] = bboxes
                result["scores"] = scores
                result["original_image"] = sahi_scaled_down_image
                if bbox_coord_wgs84 != None:
                    result["ships_long_lat"] = []
                    result["ships_lengths"] = []
                    result["ships_bbox_dimensions"] = []
            else:
                result[img[0]]["n_obj"] = 0
                result[img[0]]["bboxes"] = bboxes
                result[img[0]]["scores"] = scores
                result[img[0]]["original_image"] = sahi_scaled_down_image
            if bbox_coord_wgs84 != None:
                if bbox_coord_wgs84.get(img[0]) != None:
                    result[img[0]]["ships_long_lat"] = []
                    result[img[0]]["ships_lengths"] = []
                    result[img[0]]["ships_bbox_dimensions"] = []
            continue

        nms_result = nms(boxes=bboxes, scores=scores, iou_threshold=nms_iou_threshold)
        bboxes = bboxes.numpy() 
        bboxes_nms = []
        bboxes_nms = np.array([bboxes[i] for i in nms_result])
        scores_nms = np.array([scores[i] for i in nms_result])
        print(f"{len(bboxes_nms)} bboxes found in {img[0]}")

        # Output the result
        if inference_mode == "single_image":
            result = dict()
            result["n_obj"] = len(bboxes_nms)
            result["bboxes"] = bboxes_nms
            result["scores"] = scores_nms
            if output_original_image:
                result["original_image"] = sahi_scaled_down_image
        else:
            result[img[0]] = dict()
            result[img[0]]["n_obj"] = len(bboxes_nms)
            result[img[0]]["bboxes"] = bboxes_nms
            result[img[0]]["scores"] = scores_nms
            if output_original_image:
                result[img[0]]["original_image"] = sahi_scaled_down_image
        
        # Calculating the longitude and latitude of each bbox's center as will as the detected ship length in meters (if bbox_coord_wgs84 is given):coords_verified = False
        coords_verified = False
        if bbox_coord_wgs84 != None:
            if inference_mode == "single_image":
                if type(bbox_coord_wgs84) in [list, tuple]:
                    coords_verified, lon1, lat1, lon2, lat2 = verify_coords(coords=bbox_coord_wgs84, inference_mode=inference_mode)
                elif type(bbox_coord_wgs84) == dict:
                    coords_verified, lon1, lat1, lon2, lat2 = verify_coords(coords=bbox_coord_wgs84[list(bbox_coord_wgs84.keys())[0]], inference_mode=inference_mode) 
            elif bbox_coord_wgs84.get(img[0]) != None:
                coords_verified, lon1, lat1, lon2, lat2 = verify_coords(coords=bbox_coord_wgs84[img[0]], inference_mode=inference_mode) 

        if coords_verified:
            w_resized, h_resized = scaled_down_image_size
            dist_h = haversine_distance(lon1, lat1, lon1, lat2)
            dist_w = haversine_distance(lon1, lat1, lon2, lat1)
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

                if inference_mode == "single_image":
                    result["ships_long_lat"] = ships_coord
                    result["ships_lengths"] = ships_length
                    result["ships_bbox_dimensions"] = ships_bbox_dimensions
                else:
                    result[img[0]]["ships_long_lat"] = ships_coord
                    result[img[0]]["ships_lengths"] = ships_length
                    result[img[0]]["ships_bbox_dimensions"] = ships_bbox_dimensions
        
        # Drawing bbox and save image
        if save_annotated_image or output_annotated_image:
            image_save_name = img[1]
            if inference_mode == "single_image":
                annotated_image = draw_bbox_torchvision(image=sahi_scaled_down_image, bboxes=bboxes_nms, scores=scores_nms,
                        lengths=result.get("ships_lengths"), ships_coords=result.get("ships_long_lat"),
                        annotations=annotations, save=save_annotated_image, image_save_name=image_save_name, output_annotated_image=output_annotated_image,
                        font=annotation_font, font_size=annotation_font_size, bbox_width=annotation_bbox_width)
                if output_annotated_image:
                    result["annotated_image"] = annotated_image
            else:
                annotated_image = draw_bbox_torchvision(image=sahi_scaled_down_image, bboxes=bboxes_nms, scores=scores_nms,
                                  lengths=result[img[0]].get("ships_lengths"), ships_coords=result[img[0]].get("ships_long_lat"),
                                  annotations=annotations, save=save_annotated_image, image_save_name=image_save_name, output_annotated_image=output_annotated_image,
                                  font=annotation_font, font_size=annotation_font_size, bbox_width=annotation_bbox_width)
                if output_annotated_image:
                    result[img[0]]["annotated_image"] = annotated_image
    
    if save_annotated_image == True:
        txt_dir = path.join(output_dir, 'inference_config.txt')
        with open(txt_dir, 'w') as f:
            f.write(f"""model_or_model_path= {model_or_model_path}\nmodel_input_dim={model_input_dim}\ndevice={device}\n--------------\nconfidence_threshold={confidence_threshold}\nnms_iou_threshold={nms_iou_threshold}\nsahi_overlap_ratio={sahi_overlap_ratio}\nscale_down_factor={scale_down_factor}\nadaptive_scale_down_parameters = {adaptive_scale_down_parameters}\n--------------\nannotation_font={annotation_font}\nannotation_font_size={annotation_font_size}\nannotation_bbox_width={annotation_bbox_width}""")

    del model
    return result


# remaining modules: 
# 2- csv or json result saving module [image hyperlink - time inteval - bbox coords - n_obj - lengths sorted dscending]
# 3- API module: takes bbox of an area - splitting it to allowable areas (sentinel-gub constraints) - download images of
# area in given time period [maybe splitting time into possible time intervals] and request images of all those from sentinel 
# api and run inference on all of them. result saved for each time interval and bbox coords [hyperlink the images in csv]
