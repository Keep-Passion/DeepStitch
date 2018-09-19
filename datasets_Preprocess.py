import cv2
import glob
import random
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import skimage.io as io
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image
import matplotlib.pyplot as plt

trainRatio = 0.7    # There are 70% imgaes in dataset are train examples
lowCropRatio = 0.6
highCropRatio = 0.9

def graffitiToCSV(inputAddress, outputAddress, isFixedLength=True, fixedLength=224, overlapRatio = [0.2,0.8], cropRatio = [0.6, 0.9]):
    ''' 对graffiti数据集做处理（http://kahlan.eps.surrey.ac.uk/featurespace/web/data.htm）'''
    originalAddress = inputAddress
    imagesAddress = outputAddress
    trainImageList = []; trainDrowList = []; trainDcolList = []
    testImageList = [];  testDrowList = [];  testDcolList = []

    subFolders = os.listdir(originalAddress)
    for folder in subFolders:
        imageAddressList = glob.glob(originalAddress + folder + "\\*.png")
        for imageAddress in imageAddressList:
            image = cv2.imread(imageAddress)
            imageName = "graffiti_" + folder + "_"+imageAddress.split("\\")[-1].split(".")[0]

            tempDir = "_1"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=0, isFixedLength=isFixedLength, fixedLength=fixedLength, overlapRatio = overlapRatio, cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir +  "\\"):
                os.makedirs(imagesAddress + imageName + tempDir +  "\\")
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir); trainDrowList.append(drow); trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);  testDrowList.append(drow);  testDcolList.append(dcol)

            # print("imageName: "+imageName + tempDir +", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("1", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_2"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=1, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir +  "\\"):
                os.makedirs(imagesAddress + imageName + tempDir +  "\\")
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir); trainDrowList.append(drow); trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);  testDrowList.append(drow);  testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("2", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_3"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=2, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir +  "\\"):
                os.makedirs(imagesAddress + imageName + tempDir +  "\\")
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir); trainDrowList.append(drow); trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);  testDrowList.append(drow);  testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("3", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_4"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=3, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir +  "\\"):
                os.makedirs(imagesAddress + imageName + tempDir +  "\\")
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "\\" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir); trainDrowList.append(drow); trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);  testDrowList.append(drow);  testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("4", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

    trainDict = {'imageName': trainImageList, 'drow': trainDrowList, 'dcol': trainDcolList}
    testDict =  {'imageName': testImageList,  'drow': testDrowList,  'dcol': testDcolList}
    trainDF = pd.DataFrame(trainDict, columns=['imageName', 'drow', 'dcol'])
    testDF = pd.DataFrame(testDict, columns=['imageName', 'drow', 'dcol'])
    trainDF.to_csv(".\\datasets\\graffiti\\fromCSV\\train_fixed224.csv")
    testDF.to_csv(".\\datasets\\graffiti\\fromCSV\\val_fixed224.csv")

def getStitchByOffset(images, offset):
    ''' 根据offset生成拼接结果'''
    imageA = images[0]
    imageB = images[1]
    assert imageB.shape == imageA.shape
    h,w,c = imageA.shape
    drow = offset[0]
    dcol = offset[1]
    if drow >= 0 and dcol >= 0:
        result = np.zeros((h + drow, w + dcol, 3), imageB.dtype)
        result[0:h, 0:w, :] = imageA
        result[drow: h + drow, dcol:w + dcol, :] = imageB
    elif drow < 0 and dcol >= 0:
        result = np.zeros((h - drow, w + dcol, 3), imageB.dtype)
        result[-1 * drow: h - drow, 0:w, :] = imageA
        result[0: h, dcol: w + dcol, :] = imageB
    elif drow >= 0 and dcol < 0:
        result = np.zeros((imageB.shape[0] + drow, imageA.shape[1] - dcol, 3), imageB.dtype)
        result[0: h, -1 * dcol: w - dcol, :] = imageA
        result[drow: h + drow, 0:w, :] = imageB
    elif drow < 0 and dcol < 0:
        result = np.zeros((h - drow, w - dcol, 3), imageB.dtype)
        result[-1 * drow: h - drow, -1 * dcol: w - dcol, :] = imageA
        result[0:h, 0:w, :] = imageB
    return result

def cropSubImages(image, direction = 0, isFixedLength=False, fixedLength=224, overlapRatio = [0.2,0.8], cropRatio = [0.6, 0.9]):
    ''' 对图像在指定方向上进行裁剪'''
    h, w, c = image.shape
    rowRatio = random.uniform(cropRatio[0], cropRatio[1])
    colRatio = random.uniform(cropRatio[0], cropRatio[1])
    assert direction in [0, 1, 2, 3], 'The direction must in [0, 1, 2, 3]'
    if isFixedLength:
        assert h >= 2 * fixedLength and w >= 2 * fixedLength, 'The lenght of image must larger than two times of fixedLength'
        if direction == 0:  # The second image locate at right and bottom of the first one
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1))
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1))
            start_xB = start_xA + int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA + int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        elif direction == 1:  # The second image locate at right and above of the first one
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1))
            start_xB = start_xA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA + int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        elif direction == 2:    # The second image locate at left and bottom of the first one
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1))
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_xB = start_xA + int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        elif direction == 3:    # The second image locate at left and above of the first one:
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_xB = start_xA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        imageA = image[start_xA: start_xA + 224, start_yA: start_yA + 224, :]
        imageB = image[start_xB: start_xB + 224, start_yB: start_yB + 224, :]
        drow = start_xB - start_xA
        dcol = start_yB - start_yA
    else:
        if direction == 0:     # The second image locate at right and bottom of the first one
            imageA = image[0: int(h * rowRatio), 0: int(w * colRatio), :]
            imageB = image[h - int(h * rowRatio): h, w - int(w * colRatio): w, :]
            drow = h - int(h * rowRatio)
            dcol = w - int(w * colRatio)
        elif direction == 1:    # The second image locate at right and above of the first one
            imageA = image[h - int(h * rowRatio): h,  0: int(w * colRatio), :]
            imageB = image[0: int(h * rowRatio), w - int(w * colRatio): w, :]
            drow = -1 * (h - int(h * rowRatio))
            dcol = w - int(w * colRatio)
        elif direction == 2:    # The second image locate at left and bottom of the first one
            imageA = image[0: int(h * rowRatio), w - int(w * colRatio): w, :]
            imageB = image[h - int(h * rowRatio): h,  0: int(w * colRatio), :]
            drow = h - int(h * rowRatio)
            dcol = -1 * (w - int(w * colRatio))
        elif direction == 3:    # The second image locate at left and above of the first one
            imageB = image[0: int(h * rowRatio), 0: int(w * colRatio), :]
            imageA = image[h - int(h * rowRatio): h, w - int(w * colRatio): w, :]
            drow = -1 * (h - int(h * rowRatio))
            dcol = -1 * (w - int(w * colRatio))
    return imageA, imageB, drow, dcol

class StitchDataset(Dataset):
    ''' 建立拼接数据集 '''
    def __init__(self, isFromCSV=True, csvFile="", rootDir="", transform=None, isFixedLength=True, fixedLength=224, overlapRatio = [0.2,0.8], cropRatio = [0.6, 0.9]):
        self.isFromCSV = isFromCSV
        self.rootDir = rootDir
        if self.isFromCSV:
            self.groundTruth = pd.read_csv(csvFile)
        else:
            self.imagesList = glob.glob(self.rootDir + "*.jpg")
            self.transform = transform
            self.isFixedLength = isFixedLength
            self.fixedLength = fixedLength
            self.overlapRatio = overlapRatio
            self.cropRatio = cropRatio

    def __len__(self):
        if self.isFromCSV:
            return len(self.groundTruth)
        else:
            return len(self.imagesList)

    def __getitem__(self, idx):
        if self.isFromCSV:
            image_name = self.groundTruth.iloc[idx, 1]
            image_address = os.path.join(os.path.join(self.rootDir, image_name), image_name)
            imageA = io.imread(image_address + "_A.jpg")
            imageB = io.imread(image_address + "_B.jpg")
            offset = np.expand_dims(self.groundTruth.iloc[idx, 2:].astype('float'), axis=0)
            imageA = imageA.transpose((2, 0, 1))
            imageB = imageB.transpose((2, 0, 1))
            sample = {'imageA': torch.from_numpy(imageA).float(), 'imageB': torch.from_numpy(imageB).float(),
                      'offset': torch.from_numpy(offset).float()}
        else:
            image_name = self.imagesList[idx].split("\\")[-1].split(".")[0]
            image = io.imread(self.imagesList[idx])
            image = np.array(self.transform(PIL.Image.fromarray(image)))
            cropDirection = np.random.randint(0, 4)
            imageA, imageB, drow, dcol = cropSubImages(image, direction=cropDirection, isFixedLength=self.isFixedLength, fixedLength=self.fixedLength)
            stitchedImage = getStitchByOffset([imageA, imageB], [drow, dcol])
            # cv2.imshow("111", stitchedImage)
            # cv2.waitKey(0)
            offset = np.array([drow, dcol])
            imageA = imageA.transpose((2, 0, 1))
            imageB = imageB.transpose((2, 0, 1))
            sample = {'imageA': torch.from_numpy(imageA).float(), 'imageB': torch.from_numpy(imageB).float(),
                  'offset': torch.from_numpy(offset).float()}

        return sample

if __name__ == '__main__':
    # dataload Test
    dataTransform = transforms.Compose([
        transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]),
        transforms.RandomChoice([
            transforms.RandomRotation((90, 90), expand=True),
            transforms.RandomRotation((180, 180), expand=True),
            transforms.RandomRotation((270, 270), expand=True),
        ])
    ])
    # stitchDataset = StitchDataset(isFromCSV=False, rootDir=".\\datasets\\graffiti\\images\\train\\",
    #                               isFixedLength=True, fixedLength=224, transform=dataTransform,
    #                               overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9])
    stitchDataset = StitchDataset(isFromCSV=True, rootDir=".\\datasets\\graffiti\\fromCSV\\images_fixed\\",
                                  csvFile=".\\datasets\\graffiti\\fromCSV\\train_fixed224.csv",)
    for i in range(len(stitchDataset)):
        sample = stitchDataset[i]
        print('Sample: imageA size: '
              + str(sample['imageA'].size())
              + ", imageB size: "
              + str(sample['imageB'].size())
              + ", offset:" + str(sample['offset']))

    # graffiti analysis
    # inputAddress = ".\\datasets\\graffiti\\original\\"
    # outputAddress = ".\\datasets\\graffiti\\fromCSV\\images_fixed\\"
    # graffitiToCSV(inputAddress, outputAddress, isFixedLength=True, fixedLength=224, overlapRatio=[0.2, 0.8],
    #               cropRatio=[0.6, 0.9])
    # df = pd.read_csv(".\\datasets\\graffiti\\fromCSV\\train_fixed224.csv")
    # imagesAddress = ".\\datasets\\graffiti\\fromCSV\\images_fixed\\"
    # for i in range(len(df)):
    #     imageA = cv2.imread(imagesAddress + df['imageName'][i] + "\\" + df['imageName'][i] + "_A.jpg")
    #     imageB = cv2.imread(imagesAddress + df['imageName'][i] + "\\" + df['imageName'][i] + "_B.jpg")
    #     drow = df['drow'][i]
    #     dcol = df['dcol'][i]
    #     stitchedImages = getStitchByOffset([imageA, imageB], [drow, dcol])
    #     cv2.imshow("result", stitchedImages)
    #     cv2.waitKey(0)
