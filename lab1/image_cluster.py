# WIP!
from FC_Means import my_FCMeans
import numpy as np
import scipy.ndimage as sci_img
import matplotlib.pyplot as plt

image = sci_img.imread('lab1\\ImagensTeste\\photo001.jpg', mode='RGB')
# image = np.squeeze(image, axis=0)

new_img = []
for row in image:
  for pixel in row:
    new_img.append(pixel)
new_img=np.array(new_img)

K = 5

i = 1
epc_chp = []
while i > 0:
  try:
    epc, indexes, centers, itera = my_FCMeans(new_img,K)
    print(itera)
    i -= 1
    epc_chp.append(epc)
  except ZeroDivisionError as err:
    print(err)


print('done')