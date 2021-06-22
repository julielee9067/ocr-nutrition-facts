import csv
import logging
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

from .preprocessor import convert_image_to_bw


def preprocess_for_ocr(img, saved_location=None, save=False, enhance=1):
    """
    @param img: image to which the pre-processing steps being applied
    """
    if enhance > 1:
        img = Image.fromarray(img)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(enhance)
        img = np.asarray(img)

    img = convert_image_to_bw(img)
    _, bw_copy = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    if save:
        cv2.imwrite(str(saved_location), img)
        logging.info(f"preprocessed image saved: {saved_location}")

    return img


def ocr(img, lang, oem=1, psm=3):
    """
    @param img: The image to be OCR'd
    @param oem: for specifying the type of Tesseract engine( default=1 for LSTM OCR Engine)
    """
    config = "-l {lang} --oem {oem} --psm {psm}".format(lang=lang, oem=oem, psm=psm)

    try:
        img = Image.fromarray(img)
        text = pytesseract.image_to_string(img, config=config)
        return text
    except Exception as e:
        logging.info(e)
        return ""


if __name__ == "__main__":

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to the input image")
    # args = vars(ap.parse_args())

    filename = "0044000028015_2"
    # filename = "0044738018340_2"
    image = "testing/demo/{}.jpg".format(filename)

    with open(os.path.join("testing/results", "res_{}.txt".format(filename)), "r") as f:
        c = csv.reader(f, delimiter=",")
        result = []
        for row in c:
            result.append(tuple(map(int, row)))
        coordinates_list = tuple(result)
