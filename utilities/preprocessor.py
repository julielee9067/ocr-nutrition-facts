import logging
from pathlib import Path

import cv2
import extcolors
import numpy as np
from numpy import ndarray
from PIL import Image

logging.root.setLevel(logging.INFO)


def convert_image_to_bw(image: ndarray) -> ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def find_contours(image: ndarray, bw_copy: ndarray) -> ndarray:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(image.shape, dtype=np.uint8)
    mask1 = np.zeros(bw_copy.shape, dtype=np.uint8)
    wb_copy = cv2.bitwise_not(bw_copy)
    new_bw = np.zeros(bw_copy.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y : y + h, x : x + w] = 0
        area = cv2.contourArea(contours[idx])
        aspect_ratio = float(w) / h
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y : y + h, x : x + w])) / (w * h)

        # identify region of interest
        if r > 0.34 and 0.52 < aspect_ratio < 13 and area > 145.0:
            cv2.drawContours(mask1, [contours[idx]], -1, (255, 255, 255), -1)

            bw_temp = cv2.bitwise_and(
                mask1[y : y + h, x : x + w], bw_copy[y : y + h, x : x + w]
            )
            wb_temp = cv2.bitwise_and(
                mask1[y : y + h, x : x + w], wb_copy[y : y + h, x : x + w]
            )

            bw_count = cv2.countNonZero(bw_temp)
            wb_count = cv2.countNonZero(wb_temp)

            if bw_count > wb_count:
                new_bw[y : y + h, x : x + w] = np.copy(bw_copy[y : y + h, x : x + w])
            else:
                new_bw[y : y + h, x : x + w] = np.copy(wb_copy[y : y + h, x : x + w])

    return new_bw


def preprocess_word_image(
    image: ndarray, saved_location: Path = None, save: bool = False
):
    # image = resize_image(image=image)
    image = convert_image_to_bw(image=image)
    _, bw_copy = cv2.threshold(image, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # bilateral filter
    blur = cv2.bilateralFilter(image, 5, 75, 75)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)

    # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("gradient", grad)
    # cv2.waitKey(0)

    # binarization
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("otsu binarization", bw)
    # cv2.waitKey(0)

    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # finding contours
    contoured_img = find_contours(image=closed, bw_copy=bw_copy)
    # cv2.imshow("closing", contoured_img)
    # cv2.waitKey(0)

    result = cv2.bitwise_not(contoured_img)

    if save:
        cv2.imwrite(str(saved_location), result)
        logging.info(f"Saved preprocessed word image: {str(saved_location)}")

    return result


def invert_image(image_name: Path, saved_location: Path):
    image = cv2.imread(str(image_name))
    colors, pixel_count = extcolors.extract_from_path(str(image_name))
    is_text_white = colors[0][0][0] < 127
    if len(colors) > 1 and is_text_white:
        is_first_color_text = colors[1][1] / colors[0][1] < 0.65
        if is_first_color_text:
            image = cv2.bitwise_not(image)
            image = Image.fromarray(image)
            image = np.array(image)
            logging.info(f"Saved inverted image: {str(saved_location)}")

    cv2.imwrite(str(saved_location), image)

    return image


if __name__ == "__main__":
    test_folder = Path("error")
    for file in test_folder.glob("*"):
        invert_image(
            image_name=file, saved_location=Path("error_result").joinpath(file.name)
        )
