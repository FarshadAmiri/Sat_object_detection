from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
from sahi.slicing import slice_image
from imageutils import resize_img
from PIL import Image
import numpy as np
import torch


def sahi_slicing_inference(image_or_dir, model='models/best_model.pth', scale_down_factor=1, model_input_dim=768,
                           device='cpu', confidence_threshold = 0.9, overlap_ratio=0.2, output_scaled_down_image=False):
    
    if type(image_or_dir) == str:   # if image's path is passed into the function
        image = Image.open(image_or_dir)
    elif type(image_or_dir) == np.ndarray:
        image = Image.fromarray(np.uint8(image_or_dir)).convert('RGB')
    elif type(image_or_dir) != Image.Image:
        raise TypeError("image_or_dir should be whether a np.ndarray or PIL.Image.Image or a directory string")
    else:
        image = image_or_dir
    w, h = image.size   # PIL: w,h    Numpy: h, w, _
    resized_image = resize_img(image= image, height= h//scale_down_factor, width=w//scale_down_factor)
    resized_image = Image.fromarray(np.uint8(resized_image)).convert('RGB')

    if type(model) == str:
        model = torch.load(model, map_location = device)

        model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=confidence_threshold,
        image_size=model_input_dim,
        device=device, # or "cuda:0"
        load_at_init=True,)

    prediction = get_sliced_prediction(
    resized_image,
    model,
    slice_height=model_input_dim,
    slice_width=model_input_dim,
    overlap_height_ratio=overlap_ratio,
    overlap_width_ratio=overlap_ratio,)

    bboxes = [prediction.object_prediction_list[i].bbox for i in range(len(prediction.object_prediction_list))]
    bboxes = torch.Tensor([[i.minx, i.miny, i.maxx, i.maxy] for i in bboxes])
    # if bboxes.dim() == 1:
    scores = torch.Tensor([prediction.object_prediction_list[i].score.value for i in range(len(prediction.object_prediction_list))])
    # print("len(prediction.object_prediction_list): ", len(prediction.object_prediction_list))
    # if len(prediction.object_prediction_list) == 0:
    #     bboxes = torch.empty(3, 4, 5)
    #     scores = torch.empty(3, 4, 5)
    #     print("Done bboxeeeees")
    #     print(bboxes)
    print("bboxes.dim(): ", bboxes.dim())
    n_obj = len(scores)

    result = {"res_object": prediction,"n_obj": n_obj ,"bboxes": bboxes, "scores": scores ,"scaled_down_image_size": resized_image.size}

    if output_scaled_down_image == True:
        result["scaled_down_image"] = resized_image

    return result