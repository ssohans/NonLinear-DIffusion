# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:58:06 2020

@author: SSohan
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.ndimage.filters as flt

def prewit(imgarr):
   prewit = []
   kernelx = np.array([[1,1,1],
                       [0,0,0],
                       [-1,-1,-1]])
   kernely = np.array([[-1,0,1],
                       [-1,0,1],
                       [-1,0,1]]) 
   for i in imgarr:
      img_prewittx = cv2.filter2D(np.uint8(i),-1,kernelx)
      img_prewitty = cv2.filter2D(np.uint8(i),-1,kernely) 
      img_prewitt = img_prewittx + img_prewitty
      prewit.append(img_prewitt)
   return prewit

def sobel(imgarr):
   sobel = []
   for i in imgarr:
      img_sobelx=cv2.Sobel(i,cv2.CV_8U,1,0,ksize=3)
      img_sobely=cv2.Sobel(i,cv2.CV_8U,0,1,ksize=3)
      img_sobel = img_sobelx + img_sobely
      sobel.append(img_sobel)
   return sobel

def canny(imgarr):
   canny = []
   for i in imgarr:
      img = cv2.Canny(np.uint8(i),100,200)
      canny.append(img)
   return canny

def ploting(imgArr,imgTitles,x=1,y=1,figs=(20,25)):
   plt.figure(figsize=figs)
#   print(imgTitles)
#   x = len(imgTitles)
   for i in range(x*y):
      ax = plt.subplot(x,y,i+1)
#      if(x>0):
      ax.set_title(imgTitles[i])
      ax.axis('off')
      ax.imshow(imgArr[i],cmap='gray',interpolation='nearest')
   

def noise(img):
   gaussian = np.zeros(img.shape, np.uint8)
   mean = 0
   sigma = .05**.5
   cv2.randn(gaussian,mean,sigma)
   cv2.normalize(gaussian, gaussian, 0, 255, cv2.NORM_MINMAX, dtype=-1)
   gaussian = gaussian.astype(np.uint8)
   noisy_image = cv2.add(img, gaussian)
   return noisy_image


#base anisotropi filter by peronmallik
def anisodiff(img,it=10,gamma=.25,k=40,eq=1,step=(1.,1.)):
     # initialize output array
     img = img.astype('float32')
     imgout = img.copy()
 
     # initialize some internal variables
     deltaS = np.zeros_like(imgout)
     deltaE = deltaS.copy()
     NS = deltaS.copy()
     EW = deltaS.copy()
     gS = np.ones_like(imgout)
     gE = gS.copy()

 
     for ii in range(it):
             # calculate the differnece x to y
             deltaS[:-1,: ] = np.diff(imgout,axis=0)
             deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
             # conduction gradients (only need to compute one per dim!)
             if eq == 1:
                     gS = np.exp(-(deltaS/k)**2.)/step[0]
                     gE = np.exp(-(deltaE/k)**2.)/step[1]
             elif eq == 2:
                     gS = 1./(1.+(deltaS/k)**2.)/step[0]
                     gE = 1./(1.+(deltaE/k)**2.)/step[1]
 
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



#regularized perona-malik filter
def anisodiff2(img,it=10,gamma=.1,k=40,sigma=1,eq=1,step=(1.,1.)):
     # initialize output array
     img = img.astype('float32')
     imgout = img.copy()
 
     # initialize some internal variables
     deltaS = np.zeros_like(imgout)
     deltaE = deltaS.copy()
     NS = deltaS.copy()
     EW = deltaS.copy()
     gS = np.ones_like(imgout)
     gE = gS.copy()

 
     for ii in range(it):
       # calculate the differnece x to y
       deltaS[:-1,: ] = np.diff(imgout,axis=0)
       deltaE[: ,:-1] = np.diff(imgout,axis=1)
       
       if 0<sigma:
          deltaSf=flt.gaussian_filter(deltaS,sigma);
          deltaEf=flt.gaussian_filter(deltaE,sigma);
       else:
          deltaSf=deltaS;
          deltaEf=deltaE;
 
       # conduction gradients (only need to compute one per dim!)
       if eq == 1:
            gS = np.exp(-(deltaSf/k)**2.)/step[0]
            gE = np.exp(-(deltaEf/k)**2.)/step[1]
       elif eq == 2:
            gS = 1./(1.+(deltaSf/k)**2.)/step[0]
            gE = 1./(1.+(deltaEf/k)**2.)/step[1]
 
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

if __name__ == '__main__':
   img = cv2.imread('img/lenna.PNG',0)
   
   #adding noise to the image
   noisy_image = noise(img)
   
#   ploting([img,noisy_image],['Original Image','Noisy Image'],1,2,(15,15))
   #with noisy image
   #anisotropic
   
#   noisy_image = img;
   img5 = anisodiff(noisy_image,5)
   img10 = anisodiff(noisy_image,10)
   img15 = anisodiff(noisy_image,15)
   img20 = anisodiff(noisy_image,50)
   plt.figure(figsize=(10,12))
   plt.imshow(img20,cmap='gray')
#   imgani = [noisy_image,img5,img10,img15,img20]
#   imgtitles = ['Noisy_Image','Iteration 5','Iteration 10','Iteration 15','Iteration 20']
#   ploting(imgani,imgtitles,1,5)
   
#   ploting(canny(imgani),imgtitles,1,5)
#   ploting(sobel(imgani),imgtitles,1,5)
#   ploting(prewit(imgani),imgtitles,1,5)
   
   
   #gaussian
#   img33 = cv2.GaussianBlur(noisy_image,(3,3),0)
#   img55 = cv2.GaussianBlur(noisy_image,(5,5),0)
#   img77 = cv2.GaussianBlur(noisy_image,(7,7),0)
#   img99 = cv2.GaussianBlur(noisy_image,(9,9),0)
#   img33 = anisodiff2(img,5)
#   img55 = anisodiff2(img,10)
#   img77 = anisodiff2(img,15)
#   img99 = anisodiff2(img,20)
   
#   imgga = [noisy_image,img33,img55,img77,img99]
#   imgtitles = ['Noisy_img','3x3','5x5','7x7','9x9']
#   ploting(imgga,imgtitles,1,5)
#   
#   ploting(canny(imgga),imgtitles,1,5)
#   ploting(sobel(imgga),imgtitles,1,5)
#   ploting(prewit(imgga),imgtitles,1,5)
#   
   
   

   
