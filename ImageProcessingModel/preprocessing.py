import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import glob
import os
import sys
from scipy.spatial import distance


good = 0 #number of images with correct number of objects detected
bad = 0  #number of images with incorrect number of objects detected

#Note: only reads images from within the same folder and saves to same folder. Since didn't choose this method, not a final complete code
#Code for further identification of head direction in determiningDirection.ipynb

#loops over all images
for ind in range(160):
    img = cv2.imread('img_'+str(ind)+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make grayscale

    #thresholding to identify flies
    filt = cv2.bilateralFilter(img, 15, 5, 5)
    equ = cv2.equalizeHist(filt)
    th1 = cv2.adaptiveThreshold(filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 195, 25)
    th2 = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 195, 60)
    th = 255 - (255-th1) - (255-th2)
    
    #square kernel definitions for mophological operations for mask
    kernel_co = np.ones((15, 15), np.uint8)
    kernel_dil = np.ones((20, 25), np.uint8)
    kernel_enlarge = np.ones((2, 25), np.uint8)
    
    #morphological operations to create a mask which isolates flies from background noise
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_co)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_co)
    dilation = cv2.erode(opening, kernel_dil, 1)
    dilation = cv2.erode(dilation, kernel_enlarge, 1)
    #creates mask with 0 for background pixels
    mask = 255-dilation
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) #ensures mask is binary 0,255 image

    filt2 = cv2.bilateralFilter(img,13,30,30)

    #define circular and cross kernel for morphological operations for fly shape
    kernel = np.ones((3,3),np.uint8)
    for i in [0,2]:
        for j in [0,2]:
            kernel[i,j] = 0
            kernel[j,i] = 0

    kernel2 = np.ones((5,5),np.uint8)
    for i in [0,4]:
        for j in [0,4]:
            kernel2[i,j] = 0
            kernel2[j,i] = 0

    #combine mask and filtered image 
    thresh = 255-th1 #orig=195,25
    th_mask = 255-np.multiply(thresh,mask/255)

    #remove long vertical and horizontal lines (mask doesn't completely remove background edges/objects)
    y_size = th_mask.shape[0]
    x_size = th_mask.shape[1]
    lines = th_mask.copy()/255
    for x in range(x_size-12):
        if np.sum(lines[:,x])<(y_size*0.80): #"if % background (white) < 80%"
            lines[:,x-12:x+12] = np.ones((y_size,24))
    for y in range(y_size-12):
        if np.sum(lines[y,:])<(x_size*0.75):
            lines[y-12:y+12,:] = np.ones((24,x_size))
    lines = lines*255

    #perform morphological operations (open, close, dilate) to increase clarity of fly shape (only head and body)
    lines = cv2.morphologyEx(lines,cv2.MORPH_OPEN, kernel)
    lines = cv2.morphologyEx(lines,cv2.MORPH_CLOSE, kernel)
    lines = cv2.dilate(lines,kernel,iterations = 1)
    
    #output (save as jpg) resulting processed image (comment out for no output)
    cv2.imwrite('result'+str(ind)+'.jpg',lines)
    
    #get contours of centers
    contours, hierarchy = cv2.findContours(lines,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #define new images for further operations (same as resulting processed images)
    bound = lines.copy()
    boundc = cv2.cvtColor(bound,cv2.COLOR_GRAY2RGB) #processed image as colour image for visualization
    centroid = np.zeros((4,2))
    box_count=0 
    count=0

    #complete bouding box detection for each fly
    for i in range(1,len(contours)):
        #centroids of objects
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        #eliminate any small objects that aren't part of the fly (for example wing segments that have been separated)
        if area>400:
            #find centroid of fly
            M = cv2.moments(cnt)
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            centroid[count,0] = cX
            centroid[count,1] = cY

            #identify head direction based on distance
            dist = [distance.euclidean(centroid[count,:],pt) for pt in cnt]
            pts = [pt for pt in cnt]

            minIdx = np.argmin(dist)
            maxIdx = np.argmax(dist)
            pX = pts[maxIdx][0][0]
            pY = pts[maxIdx][0][1]

            #find rotated bounding box around object
            rect = cv2.minAreaRect(cnt)
            wh = rect[1]
            ang = rect[2]
            h = wh[0]

            #draw a line towards the direction of the head
            #if there is a wing included (i.e. fly is longer than a certain length), head direction is opposite of calculated longest distance
            if h>60: #means fly has wings shown
                cv2.line(boundc,(int(cX),int(cY)),(int(cX-(pX-cX)),int(pY-(pY-cY))),(0,0,255),3)
            else:
                cv2.line(boundc,(int(cX),int(cY)),(pX,pY),(0,0,255),3)
            
            #draw bounding box
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boundc = cv2.drawContours(boundc,[box],0,(255,0,0),2)

            #number of objects identified as flies
            box_count+=1
    
    #save image with bounding box and direction line drawn 
    cv2.imwrite('box'+str(ind)+'.jpg',boundc)

    #count number of good and bad dectections (based only on how many objects are detected)
    good += box_count==4
    bad += box_count!=4

#output success of detection
print("good to bad ratio:")
print((good-8)*1.0/(good+bad)) #subtract 8 for the images during the transition between stimulation and not
