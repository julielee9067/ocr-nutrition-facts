from typing import Any

import cv2
from numpy import ndarray


def draw(image_obj, coords, saved_location, extend_ratio=0):
    nx = image_obj.shape[1]
    ny = image_obj.shape[0]

    modified_coords = (
        int(coords[0] - extend_ratio * nx),
        int(coords[1] - extend_ratio * ny),
        int(coords[2] + extend_ratio * nx),
        int(coords[3] + extend_ratio * ny),
    )
    cv2.rectangle(
        image_obj,
        (modified_coords[0], modified_coords[1]),
        (modified_coords[2], modified_coords[3]),
        (0, 255, 0),
        2,
    )
    cv2.imwrite(saved_location, image_obj)


def crop(
    image_obj: ndarray,
    coords: Any,
    saved_location: str = None,
    extend_ratio: float = 0,
    SAVE: bool = False,
) -> ndarray:
    nx = image_obj.shape[1]
    ny = image_obj.shape[0]

    modified_coords = (
        int(coords[0] - extend_ratio * nx),
        int(coords[1] - extend_ratio * ny),
        int(coords[2] + extend_ratio * nx),
        int(coords[3] + extend_ratio * ny),
    )
    # cropped_image = image_obj.crop(modified_coords)
    modified_coords = [coord if coord >= 0 else 0 for coord in modified_coords]
    cropped_image = image_obj[
        modified_coords[1] : modified_coords[3], modified_coords[0] : modified_coords[2]
    ]
    if SAVE and cropped_image.size != 0:
        cv2.imwrite(saved_location, cropped_image)

    return cropped_image
