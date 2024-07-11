import numpy as np
from PIL import Image, ImageOps
img_size = [8,8]

img = Image.open('0.jpg')
img_gray = img.convert('L')
img_invert = ImageOps.invert(img_gray)
img_resize = img_invert.resize(img_size)
img_array = np.array(img_resize)/16
img_reshape = img_array.reshape([-1, np.prod(img_size)])

