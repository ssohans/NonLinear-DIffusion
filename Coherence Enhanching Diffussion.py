# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:55:33 2020

@author: SSohan
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
#Shock filter 
#sigma is ρ = integration scale
#str_sigma σ = structure scale

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4):
    h,w,_ = img.shape
    
    for i in range(iter_n):
#       print(i)
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#       print(gray)
       eigen = cv2.cornerEigenValsAndVecs(gray,str_sigma,3)
#       print(eigen)
       eigen = eigen.reshape(h,w,3,2) #[e1,e2],v1,v2
       x,y = eigen[:,:,1,0],eigen[:,:,1,1]
       gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
       gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
       gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
       gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
       m = gvv < 0
       ero = cv2.erode(img, None)
       dil = cv2.dilate(img, None)
       img1 = ero
       img1[m] = dil[m]
       img = np.uint8(img*(1.0 - .5) + img1*.5)
   
    return img

if __name__ == '__main__':
   img = cv2.imread('img/cry.JPG')
   sigma = int(5 * 2 + 1)
   str_sigma = int(.5 * 2 + 1)
   if(sigma%2==0):
      sigma = sigma - 1;
   if(str_sigma%2==0):
      str_sigma = str_sigma-1;
   img1 = coherence_filter(img,sigma=sigma,str_sigma=str_sigma,iter_n=5)
   plt.figure(figsize=(15,12))
   plt.subplot(121).imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   plt.subplot(122).imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))