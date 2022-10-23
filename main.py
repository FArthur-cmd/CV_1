from PIL import Image
from Utils.utils import readImageToArray, castArrayToPil, PSNR
from VNG.VNG import VNG
from time import time
import numpy as np

image = readImageToArray('./Pictures/CFA.bmp')
time_results = []
for i in range(10):
    start = time()
    vng = VNG(image)
    result = castArrayToPil(vng.process())
    end = time()
    result.save('./Pictures/Result.bmp')
    time_results.append(end-start)

print(np.mean(time_results), np.var(time_results))
print(image.shape[0]*image.shape[1] / np.mean(time_results))


print(PSNR(readImageToArray('./Pictures/Result.bmp'), readImageToArray('./Pictures/Original.bmp')))
