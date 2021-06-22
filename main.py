import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
from numpy import ndarray

from constants import FILE_PATH, IMAGE_EXTENSION, KOR_OCR_MODEL_NAME
from table_detection.detect_table_class import NutritionTableDetector
from text_detection.text_detection import load_text_model, text_detection
from text_recognition.demo import call_demo
from text_recognition.text_guesser import guess_text
from utilities.angle_corrector import rotate_image
from utilities.crop import crop, draw
from utilities.nutrient_list import make_fuzdict
from utilities.preprocessor import invert_image
from utilities.process import ocr, preprocess_for_ocr
from utilities.regex import (
    clean_string,
    fuz_check_for_label,
    get_fuz_label_from_string,
    separate_unit,
)
from utilities.spacial_map import position_definer, string_type

logging.root.setLevel(logging.INFO)


def load_model():
    """
    load trained weights for the model
    """
    global obj
    obj = NutritionTableDetector()
    logging.info("Weights Loaded")


def get_bbx_coords(image: Any, boxes: List) -> Union:
    width = image.shape[1]
    height = image.shape[0]

    ymin = boxes[0][0][0] * height
    xmin = boxes[0][0][1] * width
    ymax = boxes[0][0][2] * height
    xmax = boxes[0][0][3] * width

    return (xmin, ymin, xmax, ymax)


def deskew_image(image: ndarray, deskewed_image_path: Path) -> ndarray:
    return rotate_image(image, deskewed_image_path)


def create_dirs():
    for path in FILE_PATH.values():
        path.mkdir(exist_ok=True)
        path.joinpath("eng").mkdir(exist_ok=True)
        path.joinpath("kor").mkdir(exist_ok=True)
    logging.info("Created directories")


def draw_text_detection_bbox(
    draw_img_path: Path, index: int, blob_coord: List, init_path: Path
) -> None:
    draw_img = (
        cv2.imread(str(draw_img_path)) if index != 0 else cv2.imread(str(init_path))
    )
    draw(
        image_obj=draw_img,
        coords=blob_coord,
        saved_location=str(draw_img_path),
        extend_ratio=0.005,
    )
    logging.info(f"Drew boxes for the image and saved: {str(draw_img_path)}")


def create_file_name(original_name: str, index: int) -> str:
    name, ext = os.path.splitext(original_name)
    return f"{name}_{index}.png"


def crop_text_box(image: ndarray, blob_coord: List, word_img_location: Path = None):
    return crop(
        image_obj=image,
        coords=blob_coord,
        extend_ratio=0.005,
        saved_location=str(word_img_location),
        SAVE=True,
    )


def save_unprocessed_image(
    img_path: Path, directory_name: str, unprocessed_img: ndarray
):
    unprocessed_img_location = (
        FILE_PATH["results"].joinpath(directory_name).joinpath(img_path.name)
    )
    unprocessed_text_blob_list = text_detection(
        img_path=FILE_PATH["results"].joinpath(directory_name).joinpath(img_path.name)
    )
    unprocessed_text_location_list = list()

    unprocessed_created_directory = FILE_PATH["unprocessed_text_detection"].joinpath(
        f"{directory_name}/{img_path.stem}"
    )
    unprocessed_created_directory.mkdir(exist_ok=True)
    unprocessed_img_saved_location = (
        FILE_PATH["unprocessed_text_detection_draw"]
        .joinpath(directory_name)
        .joinpath(img_path.name)
    )
    empty_folder(unprocessed_created_directory)

    for index, blob_coord in enumerate(unprocessed_text_blob_list):
        draw_text_detection_bbox(
            draw_img_path=unprocessed_img_saved_location,
            index=index,
            blob_coord=blob_coord,
            init_path=unprocessed_img_location,
        )
        unprocessed_word_img_saved_location = unprocessed_created_directory.joinpath(
            create_file_name(Path(img_path).name, index)
        )
        word_image = crop_text_box(
            image=unprocessed_img,
            blob_coord=blob_coord,
            word_img_location=unprocessed_word_img_saved_location,
        )
        new_location = proceed_ocr(
            word_image=word_image,
            saved_location=unprocessed_created_directory.joinpath(
                create_file_name(original_name=img_path.name, index=index)
            ),
            directory_name=directory_name,
            blob_coord=blob_coord,
            unprocessed=True,
        )
        if new_location is not None:
            unprocessed_text_location_list.append(new_location)

    return unprocessed_text_location_list


def empty_folder(path: Path):
    for file in path.glob("*"):
        file.unlink(missing_ok=True)


def proceed_ocr(
    word_image: ndarray,
    saved_location: Path,
    directory_name: str,
    blob_coord: List,
    unprocessed: bool = False,
) -> Optional:
    new_location = None
    if not unprocessed:
        word_image = preprocess_for_ocr(
            img=word_image, saved_location=saved_location, save=True
        )
    if word_image.size != 0:
        word_image = invert_image(
            image_name=saved_location, saved_location=saved_location
        )

        if "eng" == directory_name:
            text = ocr(img=word_image, lang="eng", oem=1, psm=7)
        else:
            cv2.imwrite("kor_ocr_data/image.png", word_image)
            original_text = call_demo(
                saved_model=KOR_OCR_MODEL_NAME, image_folder="kor_ocr_data"
            )
            text = guess_text(text=original_text)
        if text.strip():
            center_x = (blob_coord[0] + blob_coord[2]) / 2
            center_y = (blob_coord[1] + blob_coord[3]) / 2
            box_center = (center_x, center_y)

            new_location = {
                "bbox": blob_coord,
                "text": text,
                "box_center": box_center,
                "string_type": string_type(text),
            }

    return new_location


def get_nutrition_dict(text_location_list: List):
    nutrition_dict = dict()
    for text_dict in text_location_list:
        if text_dict["string_type"] == 2:
            for text_dict_test in text_location_list:
                if (
                    position_definer(
                        text_dict["box_center"][1],
                        text_dict_test["bbox"][1],
                        text_dict_test["bbox"][3],
                    )
                    and text_dict_test["string_type"] == 1
                ):
                    text_dict["text"] = text_dict["text"].__add__(
                        " " + text_dict_test["text"]
                    )
                    text_dict["string_type"] = 0

    fuzdict = make_fuzdict("data/nutrients.txt")

    for text_dict in text_location_list:
        if text_dict["string_type"] == 0:
            text = clean_string(text_dict["text"])
            if fuz_check_for_label(text, fuzdict):
                label_name, label_value = get_fuz_label_from_string(text, fuzdict)
                nutrition_dict[label_name] = separate_unit(label_value)

    return nutrition_dict


def get_cropped_images(img_path: Path, bbox_coord: Tuple, directory_name: str):
    image = cv2.imread(str(img_path))
    cropped_image = crop(
        image_obj=image,
        coords=bbox_coord,
        saved_location=str(
            FILE_PATH["results"].joinpath(directory_name).joinpath(img_path.name)
        ),
        extend_ratio=0,
        SAVE=True,
    )
    unprocessed_cropped_image = cropped_image
    preprocessed_img_location = (
        FILE_PATH["processed"].joinpath(directory_name).joinpath(img_path.name)
    )
    processed_cropped_image = preprocess_for_ocr(
        img=cropped_image,
        saved_location=preprocessed_img_location,
        save=True,
        enhance=3,
    )
    return [processed_cropped_image, unprocessed_cropped_image]


def detect(
    img_path: Path,
    is_skewed: bool = False,
    save_drawing: bool = False,
    save_unprocessed: bool = False,
):
    image = cv2.imread(str(img_path))
    boxes, scores, classes, num = obj.get_classification(img=image)
    bbox_coord = get_bbx_coords(image=image, boxes=boxes)

    directory_name = "kor" if "kor" in str(img_path) else "eng"
    preprocessed_img_location = (
        FILE_PATH["processed"].joinpath(directory_name).joinpath(img_path.name)
    )
    deskewed_path = (
        FILE_PATH["deskewed"].joinpath(directory_name).joinpath(img_path.name)
    )

    cropped_images = get_cropped_images(
        img_path=img_path, bbox_coord=bbox_coord, directory_name=directory_name
    )
    cropped_image = cropped_images[0]
    unprocessed_cropped_image = cropped_images[1]

    if is_skewed:
        cropped_image = deskew_image(
            image=cropped_image, deskewed_image_path=deskewed_path
        )
        text_blob_list = text_detection(img_path=deskewed_path)
        draw_img_init_path = deskewed_path
    else:
        text_blob_list = text_detection(img_path=preprocessed_img_location)
        draw_img_init_path = preprocessed_img_location

    text_location_list = list()
    created_dir_path = (
        FILE_PATH["text_detection"].joinpath(directory_name).joinpath(img_path.stem)
    )
    created_dir_path.mkdir(exist_ok=True)
    empty_folder(path=created_dir_path)

    for index, blob_coord in enumerate(text_blob_list):
        if save_drawing:
            draw_text_detection_bbox(
                draw_img_path=FILE_PATH["text_detection_draw"]
                .joinpath(directory_name)
                .joinpath(img_path.name),
                index=index,
                blob_coord=blob_coord,
                init_path=draw_img_init_path,
            )

        word_image = crop_text_box(
            image=cropped_image,
            blob_coord=blob_coord,
            word_img_location=created_dir_path.joinpath(
                create_file_name(original_name=img_path.name, index=index)
            ),
        )
        new_location = proceed_ocr(
            word_image=word_image,
            saved_location=created_dir_path.joinpath(
                create_file_name(original_name=img_path.name, index=index)
            ),
            directory_name=directory_name,
            blob_coord=blob_coord,
        )
        if new_location is not None:
            text_location_list.append(new_location)

    logging.info(f"Total of {len(text_location_list)} text detected")

    if save_unprocessed:
        logging.info(f"Saving unprocessed img: {img_path.name}")
        save_unprocessed_image(
            img_path=img_path,
            directory_name=directory_name,
            unprocessed_img=unprocessed_cropped_image,
        )

    return get_nutrition_dict(text_location_list=text_location_list)


def compare_original_and_deskewed(img_path: Path):
    # detect(
    #     img_path=img_path, is_skewed=False, save_drawing=True, save_unprocessed=False
    # )
    detect(img_path=img_path, is_skewed=True, save_drawing=True, save_unprocessed=False)
    # return original if len(original.keys()) > len(deskewed.keys()) else deskewed


def main():
    load_model()
    load_text_model()
    create_dirs()

    # for image_name in FILE_PATH["resources"].joinpath("kor_test_images").glob("*"):
    #     if image_name.name.lower().endswith(IMAGE_EXTENSION):
    #         nutrition_dict = compare_original_and_deskewed(image_name)
    #         print(nutrition_dict)

    for image_name in FILE_PATH["resources"].joinpath("eng_test_images").glob("*"):
        if image_name.name.lower().endswith(IMAGE_EXTENSION):
            nutrition_dict = compare_original_and_deskewed(image_name)
            print(nutrition_dict)


if __name__ == "__main__":
    main()
