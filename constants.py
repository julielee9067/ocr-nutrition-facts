from pathlib import Path

KOR_OCR_MODEL_NAME = "models/iter_57000.pth"
KOR_OCR_SIMILARITY = 0.5
TEXT_DETECTION_MODEL_NAME = "models/ctpn.pb"
TABLE_DETECTION_MODEL_NAME = "models/frozen_inference_graph.pb"
IMAGE_EXTENSION = (".png", ".jpg", ".jpeg")

BASE_PATH = Path("resources")
FILE_PATH = {
    "resources": BASE_PATH,
    "results": BASE_PATH.joinpath("results"),
    "processed": BASE_PATH.joinpath("results/processed"),
    "deskewed": BASE_PATH.joinpath("results/deskewed"),
    "text_detection": BASE_PATH.joinpath("results/text_detection"),
    "text_detection_draw": BASE_PATH.joinpath("results/text_detection/draw"),
    "unprocessed_text_detection_draw": BASE_PATH.joinpath(
        "results/text_detection/draw/unprocessed"
    ),
    "unprocessed_text_detection": BASE_PATH.joinpath(
        "results/text_detection/unprocessed"
    ),
}

KOR_ANSWER_DICT = {
    "calories": [
        "열량",
    ],
    "fat": ["지방"],
    "protein": ["단백질"],
    "carbohydrate": ["탄수화물"],
    "trans fat": [
        "트랜스지방",
        "트랜스 지방",
    ],
    "serving size": ["1회 제공량"],
    "saturated fat": ["포화지방"],
    "cholesterol": [
        "콜레스테롤",
    ],
    "sugars": [
        "당류",
    ],
    "dietary fiber": ["식이섬유"],
    "sodium": ["나트륨"],
}

ENG_ANSWER_DICT = {
    "calories": [
        "calories",
    ],
    "fat": [
        "fat",
        "total fat",
        "lipides",
    ],
    "protein": [
        "protein",
        "protéines",
    ],
    "carbohydrate": [
        "carb",
        "carbohydrate",
        "carb.",
        "total carbohydrate",
        "total carb.",
        "total carb",
        "glucides",
    ],
    "trans fat": [
        "trans fat",
        "trans",
    ],
    "calories from fat": ["calories from fat"],
    "serving size": [
        "serving size",
        "per",
    ],
    "saturated fat": [
        "saturated fat",
        "saturated",
    ],
    "cholesterol": [
        "cholesterol",
    ],
    "sugars": [
        "sugars",
    ],
    "dietary fiber": [
        "dietary fiber",
        "fibre",
    ],
    "sodium": [
        "sodium",
    ],
}

UNIT_IGNORE_LIST = ["fl.", "fluid", "약"]
REPLACEMENT_DICT = {
    "og": "0g",
    "omg": "0mg",
    "o": "0",
}

EXCLUSION_DICT = {
    "calories": ["from"],
    "fat": ["saturated", "trans", "from"],
    "protein": [],
    "carbohydrate": [],
    "calories from fat": [],
    "trans fat": [],
    "serving size": [],
    "saturated fat": [],
    "cholesterol": [],
    "sugars": [],
    "dietary fiber": [],
    "sodium": [],
}


ENG_ANSWER_LIST = [
    "Nutrition",
    "Facts",
    "Calories",
    "Fat",
    "Carbohydrate",
    "Protein",
    "Cholesterol",
    "Sodium",
    "Serving",
]

KOR_ANSWER_LIST = [
    "영양성분",
    "영양정보",
    "열량",
    "탄수화물",
    "식이섬유",
    "당류",
    "단백질",
    "지방",
    "포화지방",
    "트랜스지방",
    "콜레스테롤",
    "나트륨",
    "비타민",
    "엽산",
    "칼슘",
    "아연",
    "철",
    "총내용량",
    "총내용량당",
    "칼로리",
]
