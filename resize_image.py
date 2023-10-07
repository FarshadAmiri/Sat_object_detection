import numpy as np
from PIL import Image
import cv2


def resize_img_dir(image_path, height=320, width=320):
    image = Image.open(image_path)
    # image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    y_border = max(height - image.shape[0], 0)
    x_border = max(width - image.shape[1], 0)
    top = y_border // 2
    bottom = y_border - top
    left = x_border // 2
    right = x_border - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))
    return image



def resize_img(image, height, width):
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    y_border = max(height - image.shape[0], 0)
    x_border = max(width - image.shape[1], 0)
    top = y_border // 2
    bottom = y_border - top
    left = x_border // 2
    right = x_border - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))
    return image