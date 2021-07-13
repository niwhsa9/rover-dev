# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:39:07 2021

@author: ashwi
"""

import cv2
#import cv2.aruco
import numpy
import time

import yaml

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

print("starting")
fsr = cv2.FileStorage("alvar_dict.yml", cv2.FileStorage.READ)

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
#arucoParams = cv2.aruco.DetectorParameters_create()

#(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
#	parameters=arucoParams)