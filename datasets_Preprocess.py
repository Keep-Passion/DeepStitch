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

trainRatio = 0.7  # There are 70% imgaes in dataset are train examples
lowCropRatio = 0.6
highCropRatio = 0.9

def superalloyDownsample():
    inputAddress = ".\\datasets\\superalloyBlade\\images_original\\"
    outputAddress = ".\\datasets\\superalloyBlade\\images_original_ds3\\"
    imageList = glob.glob(inputAddress + "*.jpg")
    for i in range(len(imageList)):
        imageName = imageList[i].split("\\")[-1]
        image = cv2.imread(imageList[i])
        row, col, _ = image.shape
        result = cv2.resize(image, (int(col/3), int(row/3)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outputAddress + imageName, result)

def superalloyBaldeToCSV():
    inputAddress = ".\\datasets\\superalloyBlade\\images_original_ds3\\"
    outputAddress = ".\\datasets\\superalloyBlade\\fromCSV\\images_notFixed\\"
    outCSVAddress = ".\\datasets\\superalloyBlade\\fromCSV\\"
    valImageList = [];
    valDrowList = [];
    valDcolList = []
    imageName = "superalloyBlade"
    offsetList = [[267, 0], [566, 1], [458, 1], [521, 1], [145, 614], [-474, -1], [-576, -1], [-415, -1], [-426, -1], [-47, 0], [-453, -1], [-538, -1]]
    imageList = glob.glob(inputAddress + "*.jpg")
    print(len(imageList))
    for i in range(len(imageList)-1):
        imageA = cv2.imread(imageList[i])
        imageB = cv2.imread(imageList[i+1])
        drow = int(offsetList[i][0])
        dcol = int(offsetList[i][1])
        valDrowList.append(drow)
        valDcolList.append(dcol)
        if not os.path.exists(outputAddress + imageName + "_" + str(i) + str("_") + str(i+1) + "\\"):
            os.makedirs(outputAddress + imageName + "_" + str(i) + str("_") + str(i+1) + "\\")
        print(outputAddress + imageName + "_" + str(i) + str("_") + str(i+1) + "\\" + imageName + "_" + str(i) + str("_") + str(i+1) + "_A.jpg")
        valImageList.append(imageName + "_" + str(i) + str("_") + str(i+1))
        cv2.imwrite(outputAddress + imageName + "_" + str(i) + str("_") + str(i+1) + "\\" + imageName + "_" + str(i) + str("_") + str(i+1) + "_A.jpg", imageA)
        cv2.imwrite(outputAddress + imageName + "_" + str(i) + str("_") + str(i + 1) + "\\" + imageName + "_" + str(i) + str("_") + str(i + 1) + "_B.jpg", imageB)
    valDict = {'imageName': valImageList, 'drow': valDrowList, 'dcol': valDcolList}
    valDF = pd.DataFrame(valDict, columns=['imageName', 'drow', 'dcol'])
    valDF.to_csv(outCSVAddress + "val_notFixed.csv")

def zirconSEMCLDownsample():
    inputAddress = "D:\\Coding_Test\\Python\\ImageStitch\\images\\zirconLarge\\"
    outputAddress = "D:\\Coding_Test\\Python\\ImageStitch\\images\\zirconLargeResized_8_INTER_AREA\\"
    subFolders = os.listdir(inputAddress)
    for folder in subFolders:
        imageList = glob.glob(inputAddress + folder + "\\*.jpg")
        if not os.path.exists(outputAddress + folder + "\\"):
            os.makedirs(outputAddress + folder + "\\")
        for i in range(len(imageList)):
            image = cv2.imread(imageList[i])
            imageName = imageList[i].split("\\")[-1]
            row, col, _ = image.shape
            result = cv2.resize(image, (int(col/8), int(row/8)), interpolation=cv2.INTER_CUBIC)
            print(outputAddress + folder + "\\" + imageName)
            cv2.imwrite(outputAddress + folder + "\\" + imageName, result)

def zirconSEMCLToCSV():
    inputAddress = "D:\\Coding_Test\\Python\\ImageStitch\\images\\zirconLargeResized_8_INTER_AREA\\"
    outputAddress = "D:\\Coding_Test\\Python\\DeepStitch\\datasets\\zirconSEMCL\\fromCSV\\"
    subFolders = os.listdir(inputAddress)
    num = 0
    for folder in subFolders:
        subFolder = inputAddress + folder +"\\"
        imageList = glob.glob(subFolder + "*.jpg")
        for i in range(len(imageList)-1):
            imageA = cv2.imread(imageList[i])
            imageB = cv2.imread(imageList[i+1])
            if not os.path.exists(outputAddress + "images_notFixed\\zirconSEMCL_" + str(num).zfill(6) + "\\"):
                os.makedirs(outputAddress + "images_notFixed\\zirconSEMCL_" + str(num).zfill(6) + "\\")
            cv2.imwrite(outputAddress + "images_notFixed\\zirconSEMCL_" + str(num).zfill(6) + "\\zirconSEMCL_" + str(num).zfill(6) + "_A.jpg", imageA)
            cv2.imwrite(outputAddress + "images_notFixed\\zirconSEMCL_" + str(num).zfill(6) + "\\zirconSEMCL_" + str(num).zfill(6) + "_B.jpg", imageB)
            num = num + 1

def mscoco2014ToCSV(inputAddress, outputAddress, isFixedLength=True, fixedLength=224, overlapRatio=[0.2, 0.8],
                    cropRatio=[0.6, 0.9]):
    originalAddress = inputAddress
    imagesAddress = outputAddress
    valImageList = [];
    valDrowList = [];
    valDcolList = []
    imageAddressList = glob.glob(originalAddress + "*.jpg")
    for i in imageAddressList:
        imageName = i.split("\\")[-1].split(".")[0]
        image = cv2.imread(i)
        print(i)
        cropDirection = np.random.randint(0, 4)
        imageA, imageB, drow, dcol = cropSubImages(image, direction=cropDirection, isFixedLength=isFixedLength,
                                                   fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                   cropRatio=cropRatio)
        if not os.path.exists(imagesAddress + imageName + "\\"):
            os.makedirs(imagesAddress + imageName + "\\")
        cv2.imwrite(imagesAddress + imageName + "\\" + imageName + "_A.jpg", imageA)
        cv2.imwrite(imagesAddress + imageName + "\\" + imageName + "_B.jpg", imageB)

        valImageList.append(imageName);
        valDrowList.append(drow);
        valDcolList.append(dcol)

    valDict = {'imageName': valImageList, 'drow': valDrowList, 'dcol': valDcolList}
    valDF = pd.DataFrame(valDict, columns=['imageName', 'drow', 'dcol'])
    valDF.to_csv("val_notfixed.csv")


def graffitiToCSV(inputAddress, outputAddress, isFixedLength=True, fixedLength=224, overlapRatio=[0.2, 0.8],
                  cropRatio=[0.6, 0.9]):
    ''' 对graffiti数据集做处理（http://kahlan.eps.surrey.ac.uk/featurespace/web/data.htm）'''
    originalAddress = inputAddress
    imagesAddress = outputAddress
    trainImageList = [];
    trainDrowList = [];
    trainDcolList = []
    testImageList = [];
    testDrowList = [];
    testDcolList = []

    subFolders = os.listdir(originalAddress)
    for folder in subFolders:
        imageAddressList = glob.glob(originalAddress + folder + "/*.png")
        for imageAddress in imageAddressList:
            image = cv2.imread(imageAddress)
            imageName = "graffiti_" + folder + "_" + imageAddress.split("/")[-1].split(".")[0]

            tempDir = "_1"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=0, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir + "/"):
                os.makedirs(imagesAddress + imageName + tempDir + "/")
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir);
                trainDrowList.append(drow);
                trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);
                testDrowList.append(drow);
                testDcolList.append(dcol)

            # print("imageName: "+imageName + tempDir +", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("1", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_2"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=1, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir + "/"):
                os.makedirs(imagesAddress + imageName + tempDir + "/")
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir);
                trainDrowList.append(drow);
                trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);
                testDrowList.append(drow);
                testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("2", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_3"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=2, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir + "/"):
                os.makedirs(imagesAddress + imageName + tempDir + "/")
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir);
                trainDrowList.append(drow);
                trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);
                testDrowList.append(drow);
                testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("3", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

            tempDir = "_4"
            imageA, imageB, drow, dcol = cropSubImages(image, direction=3, isFixedLength=isFixedLength,
                                                       fixedLength=fixedLength, overlapRatio=overlapRatio,
                                                       cropRatio=cropRatio)
            if not os.path.exists(imagesAddress + imageName + tempDir + "/"):
                os.makedirs(imagesAddress + imageName + tempDir + "/")
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_A.jpg", imageA)
            cv2.imwrite(imagesAddress + imageName + tempDir + "/" + imageName + tempDir + "_B.jpg", imageB)

            isTrain = True if np.random.random(1) < trainRatio else False
            if isTrain:
                trainImageList.append(imageName + tempDir);
                trainDrowList.append(drow);
                trainDcolList.append(dcol)
            else:
                testImageList.append(imageName + tempDir);
                testDrowList.append(drow);
                testDcolList.append(dcol)

            # print("imageName: " + imageName + tempDir + ", drow : " + str(drow) + ", dcow : " + str(dcol))
            # cv2.imshow("4", getStitchByOffset([imageA, imageB], [drow, dcol]))
            # cv2.waitKey(0)

    trainDict = {'imageName': trainImageList, 'drow': trainDrowList, 'dcol': trainDcolList}
    testDict = {'imageName': testImageList, 'drow': testDrowList, 'dcol': testDcolList}
    trainDF = pd.DataFrame(trainDict, columns=['imageName', 'drow', 'dcol'])
    testDF = pd.DataFrame(testDict, columns=['imageName', 'drow', 'dcol'])
    trainDF.to_csv("./datasets/graffiti/fromCSV/train_fixed224.csv")
    testDF.to_csv("./datasets/graffiti/fromCSV/val_fixed224.csv")


def getStitchByOffset(images, offset):
    ''' 根据offset生成拼接结果'''
    imageA = images[0]
    imageB = images[1]
    assert imageB.shape == imageA.shape
    h, w, c = imageA.shape
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


def cropSubImages(image, direction=0, isFixedLength=False, fixedLength=224, overlapRatio=[0.2, 0.8],
                  cropRatio=[0.6, 0.9]):
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
        elif direction == 2:  # The second image locate at left and bottom of the first one
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1))
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_xB = start_xA + int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        elif direction == 3:  # The second image locate at left and above of the first one:
            start_xA = int((h - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_yA = int((w - 2 * fixedLength) * np.random.uniform(0, 1) + fixedLength)
            start_xB = start_xA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
            start_yB = start_yA - int(fixedLength * np.random.uniform(overlapRatio[0], overlapRatio[1]))
        imageA = image[start_xA: start_xA + 224, start_yA: start_yA + 224, :]
        imageB = image[start_xB: start_xB + 224, start_yB: start_yB + 224, :]
        drow = start_xB - start_xA
        dcol = start_yB - start_yA
    else:
        if direction == 0:  # The second image locate at right and bottom of the first one
            imageA = image[0: int(h * rowRatio), 0: int(w * colRatio), :]
            imageB = image[h - int(h * rowRatio): h, w - int(w * colRatio): w, :]
            drow = h - int(h * rowRatio)
            dcol = w - int(w * colRatio)
        elif direction == 1:  # The second image locate at right and above of the first one
            imageA = image[h - int(h * rowRatio): h, 0: int(w * colRatio), :]
            imageB = image[0: int(h * rowRatio), w - int(w * colRatio): w, :]
            drow = -1 * (h - int(h * rowRatio))
            dcol = w - int(w * colRatio)
        elif direction == 2:  # The second image locate at left and bottom of the first one
            imageA = image[0: int(h * rowRatio), w - int(w * colRatio): w, :]
            imageB = image[h - int(h * rowRatio): h, 0: int(w * colRatio), :]
            drow = h - int(h * rowRatio)
            dcol = -1 * (w - int(w * colRatio))
        elif direction == 3:  # The second image locate at left and above of the first one
            imageB = image[0: int(h * rowRatio), 0: int(w * colRatio), :]
            imageA = image[h - int(h * rowRatio): h, w - int(w * colRatio): w, :]
            drow = -1 * (h - int(h * rowRatio))
            dcol = -1 * (w - int(w * colRatio))
    return imageA, imageB, drow, dcol


class StitchDataset(Dataset):
    ''' 建立拼接数据集 '''

    def __init__(self, isFromCSV=True, csvFile="", rootDir="", both_trsf=None, second_trsf=None, isFixedLength=True,
                 fixedLength=224, overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9]):
        self.isFromCSV = isFromCSV
        self.rootDir = rootDir
        if self.isFromCSV:
            self.groundTruth = pd.read_csv(csvFile)
        else:
            self.imagesList = glob.glob(self.rootDir + "*.jpg")
            self.both_trsf = both_trsf
            self.second_trsf = second_trsf
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
            offset = np.expand_dims(self.groundTruth.iloc[idx, 2:].astype('float'), axis=0).transpose()
            imageA = imageA.transpose((2, 0, 1))
            imageB = imageB.transpose((2, 0, 1))
            sample = {'imageA': torch.from_numpy(imageA).float(), 'imageB': torch.from_numpy(imageB).float(),
                      'offset': torch.from_numpy(offset).float(), 'image_name': image_name}
        else:
            image_name = self.imagesList[idx].split("/")[-1].split(".")[0]
            image = io.imread(self.imagesList[idx])
            if self.both_trsf is not None:
                image = np.array(self.both_trsf(PIL.Image.fromarray(image)))
            cropDirection = np.random.randint(0, 4)
            imageA, imageB, drow, dcol = cropSubImages(image, direction=cropDirection, isFixedLength=self.isFixedLength,
                                                       fixedLength=self.fixedLength)
            if self.second_trsf is not None:
                imageB = np.array(self.second_trsf(PIL.Image.fromarray(imageB)))
            stitchedImage = getStitchByOffset([imageA, imageB], [drow, dcol])
            # cv2.imshow("111", stitchedImage)
            # cv2.waitKey(0)
            offset = np.array([[drow], [dcol]])
            imageA = imageA.transpose((2, 0, 1))
            imageB = imageB.transpose((2, 0, 1))
            sample = {'imageA': torch.from_numpy(imageA).float(), 'imageB': torch.from_numpy(imageB).float(),
                      'offset': torch.from_numpy(offset).float(), 'image_name': image_name}

        return sample


if __name__ == '__main__':
    # dataload Test
    # both_trsf = transforms.Compose([
    #     transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]),
    #     transforms.RandomChoice([
    #         transforms.RandomRotation((90, 90), expand=True),
    #         transforms.RandomRotation((180, 180), expand=True),
    #         transforms.RandomRotation((270, 270), expand=True),
    #     ])
    # ])
    # second_trsf = transforms.Compose([transforms.ColorJitter()])
    # stitchDataset = StitchDataset(isFromCSV=False, rootDir="./datasets/graffiti/images/train/",
    #                               isFixedLength=True, fixedLength=224, both_trsf=both_trsf, second_trsf=second_trsf,
    #                               overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9])
    # # stitchDataset = StitchDataset(isFromCSV=True, rootDir="./datasets/graffiti/fromCSV/images_fixed/",
    # #                               csvFile="./datasets/graffiti/fromCSV/train_fixed224.csv",)
    # for i in range(len(stitchDataset)):
    #     sample = stitchDataset[i]
    #     print('Sample: imageA size: '
    #           + str(sample['imageA'].size())
    #           + ", imageB size: "
    #           + str(sample['imageB'].size())
    #           + ", offset:" + str(sample['offset'].size()))

    # graffiti analysis
    # inputAddress = "./datasets/graffiti/original/"
    # outputAddress = "./datasets/graffiti/fromCSV/images_fixed/"
    # graffitiToCSV(inputAddress, outputAddress, isFixedLength=True, fixedLength=224, overlapRatio=[0.2, 0.8],
    #               cropRatio=[0.6, 0.9])
    # df = pd.read_csv("./datasets/graffiti/fromCSV/train_fixed224.csv")
    # imagesAddress = "./datasets/graffiti/fromCSV/images_fixed/"
    # for i in range(len(df)):
    #     imageA = cv2.imread(imagesAddress + df['imageName'][i] + "/" + df['imageName'][i] + "_A.jpg")
    #     imageB = cv2.imread(imagesAddress + df['imageName'][i] + "/" + df['imageName'][i] + "_B.jpg")
    #     drow = df['drow'][i]
    #     dcol = df['dcol'][i]
    #     stitchedImages = getStitchByOffset([imageA, imageB], [drow, dcol])
    #     cv2.imshow("result", stitchedImages)
    #     cv2.waitKey(0)

    # coco analysis
    # inputAddress = "./datasets/MSCOCO2014/images/val/"
    # outputAddress = "./datasets/MSCOCO2014/fromCSV/images_notfixed/"
    # mscoco2014ToCSV(inputAddress, outputAddress, isFixedLength=False)

    zirconSEMCLToCSV()