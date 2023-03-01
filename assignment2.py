import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('query.jpg', 0)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
bf = cv2.BFMatcher()
dist = []
dir = 'tiny_data/'
for i in range(1,9):
    tmp = str(i)
    img2 = cv2.imread(dir + tmp + '.jpg', 0)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    sum=sum(x.distance for x in matches)
    dist.append(sum)

mini=min(dist)
index=dist.index(mini)
print(str(mini))