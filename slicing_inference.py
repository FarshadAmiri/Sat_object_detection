from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
from sahi.slicing import slice_image
from imageutils import resize_img_dir, resize_img
from PIL import Image
import numpy as np
import torch


def sahi_slicing_inference(image, model_dir='models/best_model.pth', scale_down_factor=2, model_input_dim=768, device='cpu', confidence_threshold = 0.9, overlap_ratio=0.2):
    if type(image) == str:   # if image's path is passed into the function
        image = Image.open(image)    
    w, h = image.size   # PIL: w,h    Numpy: h, w, _
    resized_image = resize_img(image= image, height= h//scale_down_factor, width=w//scale_down_factor)
    resized_image = Image.fromarray(np.uint8(resized_image)).convert('RGB')

    MODEL_TO_USE = 'models/best_model.pth'
    model = torch.load(MODEL_TO_USE, map_location = device)

    detection_model = AutoDetectionModel.from_pretrained(
    model_type='torchvision',
    model=model,
    confidence_threshold=confidence_threshold,
    image_size=model_input_dim,
    device=device, # or "cuda:0"
    load_at_init=True,)

    prediction = get_sliced_prediction(
    resized_image,
    detection_model,
    slice_height=model_input_dim,
    slice_width=model_input_dim,
    overlap_height_ratio=overlap_ratio,
    overlap_width_ratio=overlap_ratio,
    )

    bboxes = [prediction.object_prediction_list[i].bbox for i in range(len(prediction.object_prediction_list))]
    bboxes = torch.Tensor([[i.minx, i.miny, i.maxx, i.maxy] for i in bboxes])
    scores = torch.Tensor([prediction.object_prediction_list[i].score.value for i in range(len(prediction.object_prediction_list))])
    n_obj = len(scores)

    result = {"prediction":prediction ,"n_obj": n_obj ,"bboxes": bboxes, "scores": scores ,"resized_image_size": resized_image.size ,"resized_image": resized_image}
    return result