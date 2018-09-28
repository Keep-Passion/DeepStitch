import os
import glob
import cv2

originalAddress = "D:\\MyDocuments\\images_notfixed\\"


subFolders = os.listdir(originalAddress)
maxSize = 0
imageName = ""
imageShape = None
for folder in subFolders:
    imageAddressList = glob.glob(originalAddress + folder + "\\*.jpg")
    image = cv2.imread(imageAddressList[0])
    if image.size > maxSize:
        maxSize = image.size
        imageName = imageAddressList[0].split("\\")[-1]
        imageShape = image.shape
        print(imageShape)

print("************")
print(imageName)
print(maxSize)
print(imageShape)
