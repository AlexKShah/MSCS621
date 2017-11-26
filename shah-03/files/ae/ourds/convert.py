import glob
from PIL import Image
import numpy as np

size = 256, 256

for filename in glob.iglob('*.JPG'):
  print(filename)

  img = Image.open(filename).convert('L')
  img.thumbnail(size, Image.ANTIALIAS)
  img.save(filename, "JPEG")


