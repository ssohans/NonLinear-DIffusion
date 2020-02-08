# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:38:00 2020

@author: SSohan
"""

#wihtout structure tensor

import numpy as np
from matplotlib import pyplot as plt
import cv2
#import scipy.ndimage.filters as flt

def weickert(s,k):
   # m = 4
   ans = 1. - np.exp(-3.31588/((s/k)**4))
   return ans

def noise(img):
   gaussian = np.zeros(img.shape, np.uint8)
   mean = 0
   sigma = .75**.5
   cv2.randn(gaussian,mean,sigma)
   cv2.normalize(gaussian, gaussian, 0, 255, cv2.NORM_MINMAX, dtype=-1)
   gaussian = gaussian.astype(np.uint8)
   noisy_image = cv2.add(img, gaussian)
   return noisy_image
   

def anisodiff3(img,it=10,gamma=.1,k=40,sigma=1,step=(1.,1.)):
     # initialize output array
     img = img.astype('float32')
     imgout = img.copy()
     h,w = imgout.shape
     # initialize some internal variables
     deltaS = np.zeros_like(imgout)
     deltaE = deltaS.copy()
     NS = deltaS.copy()
     EW = deltaS.copy()
     gS = np.ones_like(imgout)
     gE = gS.copy()
     
     deltaSf = deltaS.copy()
     deltaEf = deltaS.copy()
     x = int(sigma*2 + 1)
     if(x%2==0):
        x = x- 1;
     for ii in range(it):
       # calculate the differnece x to y
       deltaS[:-1,: ] = np.diff(imgout,axis=0)
       deltaE[: ,:-1] = np.diff(imgout,axis=1)
       
       if 0<sigma:
#          gauss = flt.gaussian_filter(imgout,sigma)
          gauss = cv2.GaussianBlur(imgout,(x,x),0)
          deltaSf[:-1,:] = np.diff(gauss,axis=0)
          deltaEf[:,:-1] = np.diff(gauss,axis=1)
       else:
          deltaSf=deltaS;
          deltaEf=deltaE;
 
       # conduction gradients (only need to compute one per dim!)
       deltaSf[deltaSf<1] = 1
       deltaEf[deltaEf<1] = 1
       gS = weickert(deltaSf,k)
       gE = weickert(deltaEf,k)
#       print(ii)
#       for i in range(h):
#          for j in range(w):
#             if(deltaSf[i,j]==0):
#                gS[i,j] = 1
#             elif(deltaSf[i,j]>0):
#                gS[i,j] = func(deltaSf[i,j],k)
#                
#             if(deltaEf[i,j]==0):
#                gE[i,j] = 1
#             elif(deltaEf[i,j]>0):
#                gE[i,j] = func(deltaEf[i,j],k)
                
       
 
       # update matrices
       E = gE*deltaE
       S = gS*deltaS
 
       # subtract a copy that has been shifted 'North/West' by one pixel
       NS[:] = S
       EW[:] = E
       NS[1:,:] -= S[:-1,:]
       EW[:,1:] -= E[:,:-1]
 
       # update the image
       imgout += gamma*(NS+EW)

                     
     return imgout
  
   
img = cv2.imread('img/brain.PNG',0)

#img = noise(img)

img1 = anisodiff3(img,k=4,sigma=2,it=250,gamma=.25)

plt.figure(figsize=(10,12))
plt.subplot(121).imshow(img,cmap='gray')
plt.subplot(122).imshow(img1,cmap='gray')