from __future__ import print_function

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from text_detection.text_detection_class import NutritionTextDetector

from .lib.fast_rcnn.config import cfg
from .lib.fast_rcnn.test import _get_blobs
from .lib.rpn_msr.proposal_layer_tf import proposal_layer
from .lib.text_connector.detectors import TextDetector
from .lib.text_connector.text_connect_cfg import Config as TextLineCfg

sys.path.append(os.getcwd())


def load_text_model():
    """
    load trained weights for the text detection model
    """
    global obj
    obj = NutritionTextDetector()


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split("/")[-1]
    with open("data/results/" + "res_{}.txt".format(base_name.split(".")[0]), "w") as f:
        for box in boxes:
            if (
                np.linalg.norm(box[0] - box[1]) < 5
                or np.linalg.norm(box[3] - box[0]) < 5
            ):
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
            )
            cv2.line(
                img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2
            )
            cv2.line(
                img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2
            )
            cv2.line(
                img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2
            )

            min_x = min(
                int(box[0] / scale),
                int(box[2] / scale),
                int(box[4] / scale),
                int(box[6] / scale),
            )
            min_y = min(
                int(box[1] / scale),
                int(box[3] / scale),
                int(box[5] / scale),
                int(box[7] / scale),
            )
            max_x = max(
                int(box[0] / scale),
                int(box[2] / scale),
                int(box[4] / scale),
                int(box[6] / scale),
            )
            max_y = max(
                int(box[1] / scale),
                int(box[3] / scale),
                int(box[5] / scale),
                int(box[7] / scale),
            )

            line = ",".join([str(min_x), str(min_y), str(max_x), str(max_y)]) + "\r\n"
            f.write(line)

    img = cv2.resize(
        img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(os.path.join("data/results", base_name), img)


def return_blobs_tuple(boxes, scale):
    blob_list = []
    for box in boxes:
        min_x = min(
            int(box[0] / scale),
            int(box[2] / scale),
            int(box[4] / scale),
            int(box[6] / scale),
        )
        min_y = min(
            int(box[1] / scale),
            int(box[3] / scale),
            int(box[5] / scale),
            int(box[7] / scale),
        )
        max_x = max(
            int(box[0] / scale),
            int(box[2] / scale),
            int(box[4] / scale),
            int(box[6] / scale),
        )
        max_y = max(
            int(box[1] / scale),
            int(box[3] / scale),
            int(box[5] / scale),
            int(box[7] / scale),
        )

        blob_list.append((min_x, min_y, max_x, max_y))

    return tuple(blob_list)


def text_detection(img_path: Path):
    print(img_path)
    img = cv2.imread(str(img_path))
    img, scale = resize_im(
        img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE
    )
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs["data"]
        blobs["im_info"] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
        )
    cls_prob, box_pred = obj.get_text_classification(
        blobs
    )  # sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(
        cls_prob, box_pred, blobs["im_info"], "TEST", anchor_scales=cfg.ANCHOR_SCALES
    )

    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # draw_boxes(img, im_name, boxes, scale)
    return return_blobs_tuple(boxes, scale)
