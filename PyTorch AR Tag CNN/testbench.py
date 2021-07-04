# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:39:07 2021

@author: ashwi
"""

import cv2
import numpy
from cnn_mobile import *
import time

cam = cv2.VideoCapture(1)

dev = torch.device('cpu') 
model = ARTagModel(dev)
model.load("model-mobile.save")


while(True):
    # read frame from zed
    ret, frame = cam.read()
    left_right_image = numpy.split(frame, 2, axis=1)
    img = left_right_image[0]
    
    # prepare image for use 
    tfm = transforms.Compose([
                   transforms.ToTensor()
               ])
    imgTensor = tfm(img)
    imgTensor = imgTensor.unsqueeze(0)
    imgTensor.to(dev)
    
    # Run network 
    startTime = time.time()
    boxPred = model.model(imgTensor)
    totalTime = time.time() - startTime
    print(totalTime)
    
    # Get boxes from network output
    if( len(boxPred[0]["scores"]) != 0):
        for i in range(len(boxPred[0]["scores"])):
            certainty = boxPred[0]["scores"][i]
            box = boxPred[0]["boxes"][i].detach().numpy()
            if(certainty > 0.5):
                # Annotate output image 
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
            
    cv2.imshow("img", img)
    cv2.waitKey(1)