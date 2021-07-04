# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:07:38 2021

@author: ashwin
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import json

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from engine import train_one_epoch, evaluate
import utils


fig, ax = plt.subplots()

# Dataset for labeled AR Tag images
class ArTagDataset(Dataset):
    # folder for train data, path to labels
    def __init__ (self, folder, labelsPath):
        self.folder = folder        
        
        # Load labels json into dict
        with open(labelsPath) as labelsFile:
            self.labelsDict = json.load(labelsFile)
            
        # Create a dict for image_id -> labels, note image_id's are 1 indexed
        self.labelsByImageId = {}
        for label in self.labelsDict["annotations"]:
            if label["image_id"] in self.labelsByImageId:    
                self.labelsByImageId[label["image_id"]].append(label)
            else:
                self.labelsByImageId[label["image_id"]] = [label]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    # View particular item of dataloader
    def viewItem(self, idx):
        img = Image.open(self.folder + self.labelsDict["images"][idx]["file_name"]).convert("RGB")
        labels = self.labelsByImageId[self.labelsDict["images"][idx]["id"]]
        for label in labels:
            bbox = label["bbox"]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            #print(label)
            ax.add_patch(rect)
        ax.imshow(img)
    
     # View particular item of dataloader
    def drawPrediction(self, idx, prediction):
        img = Image.open(self.folder + self.labelsDict["images"][idx]["file_name"]).convert("RGB")
        labels = self.labelsByImageId[self.labelsDict["images"][idx]["id"]]
        for label in labels:
            bbox = prediction
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
            #print(label)
            ax.add_patch(rect)
        ax.imshow(img)

    def __getitem__(self, idx):
        # load image
        path = self.folder + self.labelsDict["images"][idx]["file_name"]
        img = Image.open(path).convert("RGB")
        
        boxes = []
        lbls = []
        areas = []
        iscrowd = []
        
        # get its labels and fill out target
        imgId = self.labelsDict["images"][idx]["id"]
        labels = self.labelsByImageId[imgId]
        for label in labels:
            bbox = label["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            lbls.append(1)
            areas.append(label["area"])
            iscrowd.append(False)
            
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(lbls)
        target["image_id"] = torch.tensor([imgId])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            
        return self.transform(img).to(self.device), target
    
    def __len__(self):
        return len(self.labelsDict["images"])
    

# Class with R-CNN
class ARTagModel: 
    # creates model parameters 
    def __init__(self, device):
        self.device = device
        self.dataset = ArTagDataset("dataset/", "labels.json")
        
        # Create model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=240, max_size=400)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 2
        # fully connected layers for class prediction and bounding box regression
        # in features is output of backbone
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)

    # load model params from file
    def load(self, saveFile):
        print(self.model.load_state_dict(torch.load(saveFile, map_location=self.device)))
        self.model.to(self.device)
        self.model.eval()

    # train this model
    def train(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
    
        num_epochs = 10
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            
        print("SAVING...")
        torch.save(self.model.state_dict(), "model-out.save")
        print("SAVED")
        

    # Visual validation. Take in a test set of images and run them through CNN 
    # Can be configured to either write json with detections OR write annotated
    # image with detections 
    def test(self, imageInFolder, imageWriteFolder, writeJsonFlag, writeImageFlag, showImageFlag,
             startIdx = 109, endIdx = 112):
        
        # Get files in data directory starting from the startIdx image 
        import glob, time
        files = glob.glob(imageInFolder+"/*")
        files = files[startIdx:endIdx]
        
        # Create empty json dictionary
        data = {}
        
        # Process each file
        for file in files:
            print("Processing: " + file)
            # Load and transform 
            img = Image.open(file).convert("RGB")
            tfm = transforms.Compose([
                    transforms.ToTensor()
                ])
            imgTensor = tfm(img)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor.to(self.device)
            # Run network 
            startTime = time.time()
            boxPred = self.model(imgTensor)
            totalTime = time.time() - startTime
            print(totalTime)
            # Create dictionary key
            data[file] = []
            
            # Get boxes from network output
            if( len(boxPred[0]["scores"]) != 0):
                for i in range(len(boxPred[0]["scores"])):
                    certainty = boxPred[0]["scores"][i]
                    box = boxPred[0]["boxes"][i].detach().numpy()
                    if(certainty > 0.5):
                        # Add to dictionary 
                        data[file].append(box.tolist())
                        # Annotate output image 
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(box, outline="red", width=3)
                        
            # Write the image as requested 
            if(writeImageFlag):
                img.save(imageWriteFolder+"/" + file.split("\\")[1], "JPEG")
            if(showImageFlag):
                img.show()
        if(writeJsonFlag):
            json.dump(data, open(imageWriteFolder+"/"+str("detections.json"), "w"))

# test code     
dev = torch.device('cpu') 
model = ARTagModel(dev)
model.load("model.save")
model.test("testset2", "scratch", True, True, True, 123, 130)