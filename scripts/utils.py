"""Draw predicted or ground truth boxes on input image."""
import imghdr
import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
# from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# import tensorflow.keras.preprocessing.image as img
import cv2
import math

from .keras_yolo import yolo, yolo_head

from functools import reduce

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    #image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


def filter_boxes(boxes, box_confidences, box_class_probs, threshold = 0.6):
   
    # compute box scores as P(each class | box_confidence)
    box_scores = box_confidences * box_class_probs

    # find the most probably class in each box and its probability
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

    # select boxes that have class score above threshold
    mask = box_class_scores > threshold

    # apply mask 
    scores = tf.boolean_mask(box_class_scores, mask=mask)
    boxes = tf.boolean_mask(boxes, mask=mask)
    classes = tf.boolean_mask(box_classes, mask=mask)

    return scores, boxes, classes

def nonmax_suppress(scores, boxes, classes, max_boxes=15, iou_threshold=0.6):

    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')

    non_max_suppressed_indices = tf.image.non_max_suppression(boxes, scores, max_boxes , iou_threshold=iou_threshold)
    
    scores = tf.gather(scores, non_max_suppressed_indices)
    boxes = tf.gather(boxes, non_max_suppressed_indices)
    classes = tf.gather(classes, non_max_suppressed_indices)

    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def scale_boxes(boxes, image_shape):
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = keras.backend.stack([height, width, height, width])
    image_dims = keras.backend.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def rescale_eval(outputs, image_shape=(720, 1280), max_boxes = 10, threshold=0.6, iou_threshold=0.5):

    box_xy, box_wh, box_confidences, box_class_probs = outputs

    # convert xy, wh boxes to corners
    boxes = yolo_boxes_to_corners(box_xy=box_xy, box_wh=box_wh)

    # filter out boxes with probability of detecting an object less than the threshold
    scores, boxes, classes = filter_boxes(boxes=boxes, box_confidences=box_confidences, box_class_probs=box_class_probs, threshold=threshold)

    # rescale boxes according to image size
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = nonmax_suppress(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)

    return scores, boxes, classes

def video_to_image(video_path, save_path, capture_rate=1):
    # capture_rate is number of capture per second
    cap=cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)
    count=0
    while True:
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frame_id % math.floor(frame_rate / capture_rate) == 0:
            outfile = f"{save_path}/frame{frame_id}.jpg"
            cv2.imwrite(outfile, frame)
            count+=1
    print(f"{count} images extracted from input video")

def annotate_image(model, model_image_size, image_path, anchors, class_names, threshold=0.3, iou_threshold=0.5, max_boxes=10):

    image, image_data = preprocess_image(image_path, model_image_size)

    model_output = model(image_data)

    model_output=yolo_head(model_output, anchors, len(class_names))

    out_scores, out_boxes, out_classes = rescale_eval(model_output, [image.size[1],  image.size[0]], max_boxes, threshold, iou_threshold)

    print(f'Found {len(out_boxes)} boxes for {image_path}')

    colors = get_colors_for_classes(len(class_names))

    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)

    return image

def annotate_images(model, model_image_size, input_path, output_path, anchors, class_names, threshold=0.3, iou_threshold=0.5, max_boxes=10):
    count=0

    os.makedirs(output_path, exist_ok=True)
    for image_path in [os.path.join(input_path, p) for p in os.listdir(input_path) if p.endswith('.jpg')]:
           
        image = annotate_image(model, model_image_size, image_path, anchors, class_names, threshold, iou_threshold, max_boxes)

        image.save(f'{output_path}/{os.path.basename(image_path)}', quality=100)
        count+=1
    print(f'Annotated {count} images')
    
def images_to_video(input_path, output_path, frame_rate=10):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(image_paths[0])

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video = cv2.VideoWriter(output_path, fourcc, 8, (img.shape[1], img.shape[0]))

    # writer = skvideo.io.FFmpegWriter("skvideo.mp4")
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)

    video.release()
        