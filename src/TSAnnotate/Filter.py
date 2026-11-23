import cv2
import numpy as np
from skimage.metrics import structural_similarity


def filter_similar(image1, image2, threshold=0.8):
    assert 0 < threshold < 1.0, "Similarity threshold should be between 0 and 1"
    image1 = np.array(image1)
    image2 = np.array(image2)
    # resize images
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # calculate SSIM
    score, _ = structural_similarity(gray1, gray2, full=True, data_range=255)
    if score > threshold:
        return True
    else:
        return False