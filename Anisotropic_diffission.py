# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:56:22 2020

@author: SSohan
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

def north(img):
   north = np.zeros_like(img)
   north[1:,:] = img[:-1,:]
   north[0,:] = img[0,:]
   return north - img

def south(img):
   south = np.zeros_like(img)
   south[:-1,:] = img[1:,:]
   south[-1,:] = img[-1,:]
   return south - img

def west(img):
   west = np.zeros_like(img)
   west[:,1:] = img[:,:-1]
   west[:,0] = img[:,0]
   return west - img

def east(img):
   east = np.zeros_like(img)
   east[:,:-1] = img[:,1:]
   east[:,-1] = img[:,-1]
   return east - img

def func(s,k,eq=1):
   if eq==1:
      return 1./(1.+(s/k)**2)
   return -np.exp(-(s/k)**2)

def Anisotropic(img,it=50,k=40,gamma=.25):
   
   for i in range(it):
      #north gradient
      dn = north(img)
      #south gradient
      ds = south(img)   
      #West gradient
      dw = west(img)
      #East gradient
      de = east(img)
          
      n = func(dn,k)
      s = func(ds,k)
      w = func(dw,k)
      e = func(de,k)
      
      img = img + gamma*(n*dn + s*ds + w*dw + e*de)
   
   return img
      
     
def noise(img):
   gaussian = np.zeros(img.shape, np.uint8)
   mean = 0
   sigma = .05**.5
   cv2.randn(gaussian,mean,sigma)
   cv2.normalize(gaussian, gaussian, 0, 255, cv2.NORM_MINMAX, dtype=-1)
   gaussian = gaussian.astype(np.uint8)
   noisy_image = cv2.add(img, gaussian)
   return noisy_image 
   

img = cv2.imread('img/brain.PNG',0)

img = noise(img)

img1 = img.astype('float32')/255

img1  = Anisotropic(img1,1)

plt.figure(figsize=(10,12))
plt.subplot(121).imshow(img,cmap='gray')
plt.subplot(122).imshow(img1,cmap='gray')
