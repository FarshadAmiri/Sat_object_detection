# -*- coding: utf-8 -*-
"""Utility functions for image processing."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import cv2
import io


def draw_image_with_boxes(filename, image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    plt.figure(figsize=(10,10))
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi = 300, bbox_inches='tight')
    plt.close()


def one_hot_encode(label, label_values):
    """ Converts a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes.
    """

    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """Transforms a one-hot format to a 2D array with only 1 channel where each
    pixel value is the classified class key.
    """

    x = np.argmax(image, axis = -1)

    return x


def colour_code_segmentation(image, label_values):
    """Given a 1-channel array of class keys assigns colour codes."""

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def save_fig(figname, **images):
    """Saves a list of images to disk."""

    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize = 20)
        plt.imshow(image)
    plt.savefig(f'{figname}.png')
    plt.close()



def resize_img_dir_padding(image_path, height=320, width=320):
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



def resize_img_padding(image, height, width):
    height, width = map(int, (height, width))
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


def resize_img(image, height, width):
    height, width = map(int, (height, width))
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    MAX_SIZE = (width, height)
    image.thumbnail(MAX_SIZE)
    image = np.asarray(image)
    return image


def annotated_image_numpy(image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    fig = plt.figure(figsize=(10,10))
    fig.imshow(image)

    # get the context for drawing boxes
    ax = fig.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    ax.xticks([])
    ax.yticks([])


    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=300)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    # plt.savefig(filename, dpi = 300, bbox_inches='tight')
    plt.close()

    return img_arr



def image_inference_annotation(image_or_dir, bbox_list, lengths=None, scores=None, output_img_name=None):
    if type(image_or_dir) == str:
        image = Image.open(image_or_dir)
    elif type(image_or_dir) == np.ndarray:
        image = Image.fromarray(np.uint8(image_or_dir)).convert('RGB')
    elif type(image_or_dir) != Image.Image:
        raise TypeError("image_or_dir should be whether a np.ndarray or PIL.Image.Image or a directory string")
    else:
        image = image_or_dir

    # plot the image
    # plt.figure(figsize=(10,10))
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in bbox_list:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_img_name, dpi = 300, bbox_inches='tight')
    plt.close()
    return



# def draw_bboxes(image, bbox_list, score_list, length_list, output_file_name):
#     # Convert the image to BGR format
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Iterate through the bounding boxes and draw them on the image
#     for i, bbox in enumerate(bbox_list):
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

#         # Draw the score and length on the bounding box
#         score = score_list[i]
#         length = length_list[i]
#         text = f"Score: {score}, Length: {length}"
#         cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Save the image to the output file
#     cv2.imwrite(output_file_name, image)


def draw_bboxes(image, bbox_list, score_list, output_file_name, length_list=None):
    # Convert the image to BGR format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    score_list = [float('%.2f' % elem) for elem in score_list]
    # Iterate through the bounding boxes and draw them on the image
    for i, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = map(int, bbox)
        print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw the score annotation on the bounding box
        score = score_list[i]
        cv2.putText(image, str(score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # If length_list is not None, draw the length annotation on the bounding box
        if length_list is not None:
            length = length_list[i]
            cv2.putText(image, str(length), (int((x1+x2)*0.5), int((y1+y2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 200), 2)

    # Save the image to the output file
    cv2.imwrite(output_file_name, image)

    return image