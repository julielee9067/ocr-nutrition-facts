import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict

import boto3
import cv2
import pytesseract
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from deskew import determine_skew
from google.cloud import vision
from msrest.authentication import CognitiveServicesCredentials
from numpy import ndarray

from config import (
    AZURE_OCR_ENDPOINT_URL,
    AZURE_OCR_SECRET_KEY,
    CLOVA_OCR_BASE_BODY,
    CLOVA_OCR_HEADERS,
    NAVER_APIGW_INVOKE_URL,
)
from constants import ENG_ANSWER_LIST
from utilities.angle_corrector import rotate

logging.root.setLevel(logging.INFO)


def rotate_image(file_name: Path) -> ndarray:
    image = cv2.imread(str(file_name))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(image)
    rotated = rotate(image, angle, (0, 0, 0))

    return rotated


def is_lang_eng(file_name: Path) -> bool:
    original_img = cv2.imread(str(file_name))
    rotated_image = rotate_image(file_name=file_name)
    original_extracted_text = pytesseract.image_to_string(original_img, lang="eng")
    rotated_extracted_text = pytesseract.image_to_string(rotated_image, lang="eng")
    total_text = original_extracted_text + rotated_extracted_text

    for word in ENG_ANSWER_LIST:
        if word in total_text:
            return True
    return False


def get_image_stream(file_name: Path) -> str:
    with open(file_name, "rb") as image_file:
        base64_bytes = base64.b64encode(image_file.read())

    return base64_bytes.decode("utf-8")


def call_clova_ocr_api(file_name: Path) -> Dict:
    image = {
        "format": re.sub(r"\.", "", file_name.suffix),
        "name": file_name.stem,
        "data": get_image_stream(file_name=file_name),
    }

    data = CLOVA_OCR_BASE_BODY
    data["images"] = [image]

    response = requests.post(
        url=NAVER_APIGW_INVOKE_URL, data=json.dumps(data), headers=CLOVA_OCR_HEADERS
    )
    original_dict = json.loads(response.content)
    final_dict = original_dict["images"][0]

    return final_dict


def call_azure_ocr_api(file_name: Path) -> Dict:
    computervision_client = ComputerVisionClient(
        endpoint=AZURE_OCR_ENDPOINT_URL,
        credentials=CognitiveServicesCredentials(AZURE_OCR_SECRET_KEY),
    )
    read_response = computervision_client.read_in_stream(
        open(str(file_name), "rb"), raw=True, language="en", model_version="latest"
    )
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    response_dict = {"fields": list()}

    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                response_dict["fields"].append(
                    {
                        "inferText": line.text,
                        "boundingPoly": {
                            "vertices": [
                                {
                                    "x": line.bounding_box[i],
                                    "y": line.bounding_box[i + 1],
                                }
                                for i in range(0, len(line.bounding_box), 2)
                            ]
                        },
                    }
                )

    return response_dict


def call_gcp_vision_api(file_name: Path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./config.json"
    client = vision.ImageAnnotatorClient()
    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    final_dict = {"fields": list()}

    for text in texts:
        final_dict["fields"].append(
            {
                "inferText": text.description,
                "boundingPoly": {
                    "vertices": [
                        {
                            "x": vertex.x,
                            "y": vertex.y,
                        }
                        for vertex in text.bounding_poly.vertices
                    ]
                },
            }
        )

    return final_dict


def call_aws_rekognition(file_name: Path):
    client = boto3.client("rekognition")

    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    response = client.detect_text(Image={"Bytes": content})
    final_dict = {"fields": list()}
    text_detections = response["TextDetections"]
    for text in text_detections:
        final_dict["fields"].append(
            {
                "inferText": text["DetectedText"],
            }
        )
    return final_dict


def main(file_name: Path) -> Dict:
    is_eng = is_lang_eng(file_name=file_name)
    dict_list = dict()
    time_dict = dict()

    if not is_eng:
        response_dict = call_clova_ocr_api(file_name=file_name)
        dict_list["clova"] = response_dict
    else:
        azure_start = time.time()
        azure_dict = call_azure_ocr_api(file_name=file_name)
        azure_end = time.time()
        time_dict["azure"] = str(round(azure_end - azure_start, 2))

        aws_start = time.time()
        aws_dict = call_aws_rekognition(file_name=file_name)
        aws_end = time.time()
        time_dict["aws"] = str(round(aws_end - aws_start, 2))

        gcp_start = time.time()
        gcp_dict = call_gcp_vision_api(file_name=file_name)
        gcp_end = time.time()
        time_dict["gcp"] = str(round(gcp_end - gcp_start, 2))

        dict_list = {"azure": azure_dict, "aws": aws_dict, "gcp": gcp_dict}

    for folder_name in dict_list.keys():
        try:
            with open(
                f"resources/ocr_result/{folder_name}/{file_name.stem}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(dict_list[folder_name], f, ensure_ascii=False, indent=4)
                logging.info(
                    f"Successfully created ocr file for {folder_name}/{file_name.stem}"
                )
        except Exception as e:
            logging.error(f"{folder_name}: {e}")
            pass

    time_dict["file_name"] = file_name.name
    return time_dict


if __name__ == "__main__":
    kor_file = Path("resources/kor_test_images/kor_real_8.png")
    eng_file = Path("resources/eng_test_images/eng_real_9.jpg")
    final_time_dict_list = list()

    for index, file in enumerate(Path("resources/eng_test_images").glob("*")):
        splitted_name = file.name.split("_")
        if int(re.search(r"\d+", splitted_name[2])[0]) > 10:
            final_time_dict_list.append(main(file))

    print(final_time_dict_list)
