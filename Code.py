import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt


def NonMaximalSuppression(img, radius):
    """
    take top 4% as maximum and only consider pixel as local maxima
    if it is around a pixel with a value = to any of the 4%
    """

    return img

"""
Sample Input: Input/chessboard.jpg
Test Input: Input/*
Steps:
1- Gradients in both the X and Y directions.
2- Smooth the derivative a little using gaussian (can be any other smoothing)
3- Calculate R:
	3.1 Loop on each pixel:
	3.2 Calculate M for each pixel:
		3.2.1 calculate a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2 
	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
	3.4 Calculate Response at this pixel = det-k*trace^2
	3.5 Display the result, but make sure to re-scale the data in the range 0 to 255 
4- Threshold and Non-Maximal Suppression 

"""
def harris(img, verbose=True):
    # 1- gradients in both the X and Y directions.
    Gx, Gy = utl.get_gradients_xy(img, 5)

    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    # 2- smooth the derivative a little using GaussianBlur 
    # Use 5x5 kernel with sigma=3
    #Student Code ~ 2 Lines
	
    #End of Student Code

    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04

    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
            # 3.2 Calculate M for each pixel:
            #     M = [[a11, a12],
            #          [a21, a22]]
            # where a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2

            #Student Code ~ 1 line of code
            M = None
            #End of Student Code

            # 3.3 Calculate Det_M
            # Hint: use np.linalg.det(a) or Det(a) = a11*a22 - a12*a21;
            # Student Code ~ 1 line of code
            Det_M = None
            # End of Student Code

            # 3.4 Calculate Response at this pixel = det-k*trace^2
            # where trace=a11+a22
            #Student Code ~ 1 line of code
            pass
            #End of Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255
    R = utl.rescale(R, 0, 255)
    # plt.imshow(R, cmap="gray")

    # 5- Threshold and Non-Maximal Suppression
    # Student Code ~ 2 lines of code
    pass
    pass
    # End of Student Code

    R = NonMaximalSuppression(R, 2)
    return R

img_pairs = [['check.bmp', 'check_rot.bmp'],
             ['simA.jpg', 'simB.jpg'],
             ['transA.jpg', 'transB.jpg']]
dir = 'input/'
i = 0

for img1,img2 in img_pairs:
    i += 1
    image1 = cv2.imread(dir+img1, 0)
    image2 = cv2.imread(dir+img2 , 0)
    r1 = harris(image1)
    r2 = harris(image2)
    plt.figure(i)
    plt.subplot(221), plt.imshow(image1, cmap='gray')
    plt.subplot(222), plt.imshow(image2, cmap='gray')
    plt.subplot(223), plt.imshow(r1, cmap='gray')
    plt.subplot(224), plt.imshow(r2, cmap='gray')
    plt.show()