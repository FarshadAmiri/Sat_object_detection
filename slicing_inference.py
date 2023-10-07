from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
from sahi.slicing import slice_image
from resize_image import resize_img_dir, resize_img
from PIL import Image
import numpy as np
import torch


def sahi_inference(image, model_dir='models/best_model.pth', scale_factor=2, model_input_dim=768, device='cpu', confidence_threshold = 0.7, overlap_ratio=0.2):
    device = torch.device(device)   
    if type(image) == str:
        image_dir = image
        resized_image = resize_img_dir(image_path= image_dir, height= model_input_dim*scale_factor, width=model_input_dim*scale_factor)
        resized_image = Image.fromarray(np.uint8(resized_image)).convert('RGB')
    else:
        resized_image = resize_img(image= image, height= model_input_dim*scale_factor, width=model_input_dim*scale_factor)
    
    MODEL_TO_USE = 'models/best_model.pth'
    model = torch.load(MODEL_TO_USE, map_location = device)

    detection_model = AutoDetectionModel.from_pretrained(
    model_type='torchvision',
    model=model,
    confidence_threshold=confidence_threshold,
    image_size=model_input_dim,
    device=device, # or "cuda:0"
    load_at_init=True)

    prediction = get_sliced_prediction(
    resized_image,
    detection_model,
    slice_height=model_input_dim,
    slice_width=model_input_dim,
    overlap_height_ratio=overlap_ratio,
    overlap_width_ratio=overlap_ratio,
    )

    bboxes = [prediction.object_prediction_list[i].bbox for i in range(len(prediction.object_prediction_list))]
    bboxes = [(i.minx, i.miny, i.maxx, i.maxy) for i in bboxes]
    n_obj = [len(i) for i in bboxes]

    scores = [prediction.object_prediction_list[i].score.value for i in range(len(prediction.object_prediction_list))]

    result = {"n_obj": n_obj ,"bboxes": bboxes, "scores": scores ,"resized_image_size": resized_image.shape}
    return result



