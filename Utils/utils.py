from PIL import Image
import numpy as np

def readImageToArray(path: str) :
    return np.array(Image.open(path), dtype=np.int64)

def readImageToPIL(path: str) :
    return Image.open(path)

def castPilToArray(value):
    return np.array(value, dtype=np.int64)

def castArrayToPil(value):
    return Image.fromarray(np.array(value, dtype=np.uint8))

def calculate_Y(value):
    r, g, b = value
    return 0.299 * r + 0.587 * g + 0.114 * b

def PSNR(picture, original):
    y_picture = np.apply_along_axis(calculate_Y, -1, picture)
    y_ref = np.apply_along_axis(calculate_Y, -1, original)
    return 10 * np.log10(255 ** 2 * (picture.shape[0] * picture.shape[1])/ np.sum((y_picture - y_ref) ** 2))
