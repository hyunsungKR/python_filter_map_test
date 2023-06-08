import cv2
import sys
import os
import numpy as np

strength = 2.0

diffuse = cv2.imread('C:\\Users\\LEE CHANG YOUNG\\Documents\\GitHub\\python_normalmap_test\\normalmap_sample_img\\test_image\org.jpg')

average = cv2.cvtColor(diffuse, cv2.COLOR_RGB2GRAY)

average = cv2.normalize(average.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

sobelx = cv2.Sobel(average, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(average, -1, 0, 1, ksize=3)

ones = np.ones(average.shape)

n = np.stack((strength*sobelx,strength*sobely, ones), axis=2)

norm =  np.linalg.norm(n, axis=2, keepdims=True)

n = n / norm

n = n*0.5+0.5

normalmap = np.stack((n[:, : , 2], n[:, :, 1], 1.0 - n[: ,: ,0]), axis=2)

normalmap = (normalmap * 255).astype(np.uint8)

save_path = 'C:\\Users\\LEE CHANG YOUNG\\Documents\\GitHub\\python_normalmap_test\\normalmap_sample_img\\result_image\\result1.jpg'
cv2.imwrite(save_path, normalmap)

cv2.imshow("output", normalmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

