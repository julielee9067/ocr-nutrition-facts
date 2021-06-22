import json
import re
from pathlib import Path
from typing import Dict, List

from constants import (
    ENG_ANSWER_DICT,
    EXCLUSION_DICT,
    KOR_ANSWER_DICT,
    REPLACEMENT_DICT,
    UNIT_IGNORE_LIST,
)


def get_json_dict(file_name: Path) -> Dict:
    with open(file_name) as f:
        json_dict = json.load(f)
    return json_dict


def get_nutrition_dict(file_name: Path) -> Dict:
    def contains_exclusion_word(key: str) -> bool:
        return any(word in temp_str for word in EXCLUSION_DICT[key])

    def does_not_exist_in_dict(key: str) -> bool:
        return total_dict.get(key) is None

    def ends_with_number(temp_str: str) -> bool:
        return re.search(r"\d+", temp_str.split()[-1]) and not re.search(
            r"\d+", temp_str.split()[-2]
        )

    def get_serving_size_unit() -> str:
        unit = text_list[index + 4]
        return unit + f" {text_list[index + 5]}" if unit in UNIT_IGNORE_LIST else unit

    def insert_space_in_front_of_digit(text_list: List) -> List:
        for index, word in enumerate(text_list):
            includes_number = re.search(r"\d+", word)
            if includes_number and includes_number.start() != 0:
                text_list[index] = (
                    word[: includes_number.start()]
                    + " "
                    + word[includes_number.start() :]
                )
        return text_list

    def replace_wrong_words(text_list: List) -> List:
        replacement_keys = list(REPLACEMENT_DICT.keys())
        return [
            REPLACEMENT_DICT[text_list[index]] if word in replacement_keys else word
            for index, word in enumerate(text_list)
        ]

    dict_file = get_json_dict(file_name=file_name)
    fields = dict_file.get("fields")
    text_list = " ".join(list(map(lambda x: x["inferText"].lower(), fields))).split()
    text_list = insert_space_in_front_of_digit(text_list=text_list)
    text_list = replace_wrong_words(text_list=text_list)

    total_dict = dict()
    answer_dict = KOR_ANSWER_DICT if "kor" in file_name.stem else ENG_ANSWER_DICT
    for key, values in answer_dict.items():
        for index, word in enumerate(text_list):
            temp_str = " ".join(text_list[index : index + 4])
            for similar_word in values:
                if (
                    similar_word in temp_str
                    and ends_with_number(temp_str)
                    and not contains_exclusion_word(key)
                    and does_not_exist_in_dict(key)
                ):
                    total_dict[key] = re.search(r"(\d+.\d|\d)+", temp_str.split()[-1])[
                        0
                    ]
                    if key == "serving size":
                        unit = get_serving_size_unit()
                        total_dict[key] += f" {unit}"

    print(f"{file_name.stem}: {total_dict}")
    return total_dict


if __name__ == "__main__":
    azure_dict = dict()
    file_path = Path("resources/ocr_result/azure/")
    for file in file_path.glob("*.json"):
        azure_dict[file.stem] = get_nutrition_dict(file_name=file)

    # clova_dict = dict()
    # file_path = Path("resources/ocr_result/clova/")
    # for file in file_path.glob("*.json"):
    #     clova_dict[file.stem] = get_nutrition_dict(file_name=file)

    # file_path = Path("resources/ocr_result/gcp/eng_real_10.json")
    # nut_dict = get_nutrition_dict(file_name=file_path)
