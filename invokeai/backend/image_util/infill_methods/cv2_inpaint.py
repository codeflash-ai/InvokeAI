import cv2
import numpy as np
from PIL import Image


def cv2_inpaint(image: Image.Image) -> Image.Image:
    # Prepare Image
    image_array = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Prepare Mask From Alpha Channel
    mask = image.split()[3]
    mask_array = np.array(mask)
    mask_inv = cv2.bitwise_not(mask_array)

    # Inpaint Image
    inpainted_result = cv2.inpaint(image_cv, mask_inv, 3, cv2.INPAINT_TELEA)
    inpainted_image = Image.fromarray(cv2.cvtColor(inpainted_result, cv2.COLOR_BGR2RGB))
    return inpainted_image
