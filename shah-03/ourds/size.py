import glob
from PIL import Image
import numpy as np

size = 256, 256

for filename in glob.iglob('*.jpg'):
  img = Image.open(filename)
  if (img.size[0] != 256) or (img.size[1] != 256):
    print(filename)
    print img.size
    img = img.resize(size, resample=Image.BICUBIC)
    img.save(filename, "JPEG")

