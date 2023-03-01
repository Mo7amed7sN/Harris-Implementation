import cv2
import numpy as np

def get_gradients_xy(img, ksize):
    
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1,0,ksize=ksize)
    ## Student Code ~ 1 line of code for img normalization.
    sobely = cv2.Sobel(img, cv2.CV_16S, 0,1,ksize=ksize)
	## Student Code ~ 1 line of code for img normalization
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)

    sobelx = np.uint8(sobelx)
    sobely = np.uint8(sobely)
    return sobelx, sobely


def rescale(img, min,max):
    ## Student Code ~ 2 lines of code for img normalization
    img = (img - img.min()) / float(img.max() - img.min())
    img = min + img * (max - min)
    ## End of Student Code
    return img
