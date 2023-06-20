import os
import cv2
import numpy as np
from glob import glob

input_dir = 'C:\\Users\\LEE CHANG YOUNG\\Documents\\GitHub\\python_filter_map_test\\normalmap_sample_img\\ng_carpet'
output_dir = 'C:\\Users\\LEE CHANG YOUNG\\Documents\\GitHub\\python_filter_map_test\\bandpass_normal_result_ng_carpet'

# 폴더가 존재하지 않으면 새로 만듭니다.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# glob을 사용하여 모든 jpg 이미지 파일을 불러옵니다.
img_paths = glob(os.path.join(input_dir, '*.png'))

# 각 이미지 파일에 대해 처리합니다.
for img_path in img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 히스토그램 평활화 적용
    equ = cv2.equalizeHist(img)

    # 밝기 증가
    brightness_factor = 10
    brightened_image = cv2.add(equ, np.ones_like(equ) * brightness_factor)

    # Low-pass 필터 적용 (Gaussian Blur)
    low_passed = cv2.GaussianBlur(brightened_image, (25, 25), 0)

    # High-pass 필터 적용 (Laplacian)
    laplacian = cv2.Laplacian(brightened_image, cv2.CV_64F)
    high_passed = cv2.convertScaleAbs(laplacian, alpha=2.0)

    # Bandpass 필터 결과 = 이미지 - Low-pass + High-pass
    band_passed = cv2.addWeighted(brightened_image, 1, low_passed, -0.5, 0)
    band_passed = cv2.addWeighted(band_passed, 1, high_passed, 0.5, 0)

    # Normalization
    normalized = cv2.normalize(band_passed.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Sobel Filters
    sobelx = cv2.Sobel(normalized, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(normalized, -1, 0, 1, ksize=3)

    # Create Normal Map
    ones = np.ones(normalized.shape)
    n = np.stack((sobelx, sobely, ones), axis=2)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / norm
    n = n * 0.5 + 0.5
    normal_map = np.stack((n[:, :, 2], n[:, :, 1], 1.0 - n[:, :, 0]), axis=2)
    normal_map = (normal_map * 255).astype(np.uint8)

    # Save the result
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, normal_map)