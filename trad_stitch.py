import cv2
import numpy as np
import pandas as pd
import time

class Stitcher():
    def __init__(self, stitch_mode=0, feature=0, search_ratio=0.75, offset_match=0):
        self.stitch_mode = stitch_mode      # "0" for translational mode and "1" for homography mode
        self.feature = feature              # "0" for "sift" and "1" for "surf" and "2" for "orb"
        self.search_ratio = search_ratio    # "0.75" is commonly used
        self.offset_match = offset_match    # "0" for "mode" and "1" for "ransac"


    def detectAndDescribe(self, image):
        '''
        计算图像的特征点集合，并返回该点集＆描述特征
        :param image:需要分析的图像
        :return:返回特征点集，及对应的描述特征
        '''
        if self.feature == 0:                           # "sift"
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif self.feature == 1:                         # "surf"
            descriptor = cv2.xfeatures2d.SURF_create()
        elif self.feature == 2:                         # "orb"
            descriptor = cv2.ORB_create()
            # 检测SIFT特征点，并计算描述子
        kps, features = descriptor.detectAndCompute(image, None)
        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def getOffsetByMode(self, kpsA, kpsB, matches):
        if len(matches) == 0:
            return [0, 0]
        dxList = []; dyList = [];
        for trainIdx, queryIdx in matches:
            ptA = (kpsA[queryIdx][1], kpsA[queryIdx][0])
            ptB = (kpsB[trainIdx][1], kpsB[trainIdx][0])
            # dxList.append(int(round(ptA[0] - ptB[0])))
            # dyList.append(int(round(ptA[1] - ptB[1])))
            # if int(round(ptA[0] - ptB[0])) == 0 and int(round(ptA[1] - ptB[1])) == 0:
            #     continue
            dxList.append(int(round(ptA[0] - ptB[0])))
            dyList.append(int(round(ptA[1] - ptB[1])))
        if len(dxList) == 0:
            dxList.append(0); dyList.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dxList, dyList)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))

        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        if num < 5:
            dx = 0
            dy = 0
        # print("dx = " + str(dx) + ", dy = " + str(dy) + ", num = " + str(num))
        # self.printAndWrite("  In Mode, The number of num is " + str(num) + " and the number of offsetEvaluate is "+str(offsetEvaluate))
        return [dx, dy]

    def getHomography(self, kpsA, kpsB, matches):
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        if len(matches) < 4 or kpsA.shape[0] < 4 or kpsB.shape[0] < 4:
            return np.zeros((3, 3), dtype=np.int)
        # H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
        if H is None:
            return np.zeros((3, 3), dtype=np.int)
        return H

    def evaluateByFeatureSearch(self, images, groundTrue):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuseMethod:
        :param direction: stitching direction
        :return:
        '''
        stitch_status = False
        (imageA, imageB) = images
        # get the feature points
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        if featuresA is None or featuresB is None:
            return stitch_status,1000000000
        print(" The feature num of imageA is {}".format(featuresA.shape[0]))
        print(" The feature num of imageB is {}".format(featuresB.shape[0]))

        # match the feature points
        matches = []
        if self.feature == 0 or self.feature == 1:      # For surf or sift
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
            raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
            for m in raw_matches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                if len(m) == 2 and m[0].distance < m[1].distance * self.search_ratio:
                    # 存储两个点在featuresA, featuresB中的索引值
                    matches.append((m[0].trainIdx, m[0].queryIdx))
        elif self.feature == 2:                          # For orb
            matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
            raw_matches = matcher.match(featuresA, featuresB)
            for m in raw_matches:
                matches.append((m.trainIdx, m.queryIdx))
        print(" The match num of two images is {}".format(len(matches)))


        distance = 0
        if self.stitch_mode == 0 and self.offset_match == 0:
            prd_offset = self.getOffsetByMode(kpsA, kpsB, matches)
            distance = np.linalg.norm(np.array(prd_offset) - np.array(groundTrue))
            print(" prd_offset={}, gt_offset={}, The matching result is {}".format(prd_offset, groundTrue, (np.array(prd_offset) == np.array(groundTrue)).all()))
            if (np.array(prd_offset) == np.array(groundTrue)).all():
                stitch_status = True
        elif self.stitch_mode == 0 and self.offset_match == 1:
            H = self.getHomography(kpsA, kpsB, matches)
            prd_offset = np.array([int(round(-H[1, 2])), int(round(-H[0, 2]))])
            distance = np.linalg.norm(np.array(prd_offset) - np.array(groundTrue))
            print(" prd_offset={}, gt_offset={}, The matching result is {}".format(prd_offset, groundTrue, (np.array(prd_offset) == np.array(groundTrue)).all()))
            if (np.array(prd_offset) == np.array(groundTrue)).all():
                stitch_status = True
        elif self.stitch_mode == 1 and self.offset_match == 1:
            H = self.getHomography(kpsA, kpsB, matches)
        else:
            print("Input error, No such mathcing algorithm")
        return stitch_status, distance


if __name__ == '__main__':
    # # input_address = ".\\datasets\\graffiti\\fromCSV\\images_notFixed\\"
    # # input_csv = ".\\datasets\\graffiti\\fromCSV\\val_notFixed.csv"
    # input_address = "D:\\MyDocuments\\images_notfixed\\"
    # input_csv = "D:\\MyDocuments\\val_notFixed.csv"
    # stitch_mode = 0         # "0" for translational mode and "1" for homography mode
    # feature = 2             # "0" for "sift" and "1" for "surf" and "2" for "orb"
    # search_ratio = 0.75     # "0.75" is commonly used
    # offset_match = 1        # "0" for "mode" and "1" for "ransac"
    #
    # stitcher = Stitcher(stitch_mode=stitch_mode, feature=feature, search_ratio=search_ratio, offset_match=offset_match)
    # csv_file = pd.read_csv(input_csv)
    # distance_loss = 0
    # correct_num = 0
    # time_start = time.time()
    # false_image = []
    # for idx in range(len(csv_file)):
    #     image_name = csv_file.iloc[idx, 1]
    #     print("the {} th images, Analysising {}".format(idx, image_name))
    #     local_start_time = time.time()
    #     imageA = cv2.imread(input_address + image_name + "\\" + image_name + "_A.jpg")
    #     imageB = cv2.imread(input_address + image_name + "\\" + image_name + "_B.jpg")
    #     drow = csv_file.iloc[idx, 2]
    #     dcol = csv_file.iloc[idx, 3]
    #     stitch_status, distance = stitcher.evaluateByFeatureSearch([imageA, imageB], [drow, dcol])
    #     distance_loss = distance_loss + distance
    #     if stitch_status:
    #         correct_num = correct_num + 1
    #     else:
    #         false_image.append(image_name)
    #     local_end_time = time.time()
    #     print('The duration time cost is {} s'.format(local_end_time - local_start_time))
    #     print('Now, the number of false match is {}'.format(len(false_image)))
    # time_end = time.time()
    # print('The duration time cost is {} s, and the average time cost is {}'.format(time_end - time_start, (time_end - time_start) / len(csv_file)))
    # print('The average accuracy is {} %, and the average distance loss is {}'. format(correct_num / len(csv_file) * 100, distance_loss / len(csv_file)))
    # # print("False Images: {}".format(false_image))
    # f = open('tt.txt', 'w')
    # f.write(str(false_image))
    # f.close()

    input_address = ".\\datasets\\zirconSEMCL\\fromCSV\\images_notFixed"
    stitch_mode = 0         # "0" for translational mode and "1" for homography mode
    feature = 2             # "0" for "sift" and "1" for "surf" and "2" for "orb"
    search_ratio = 0.75     # "0.75" is commonly used
    offset_match = 1        # "0" for "mode" and "1" for "ransac"

    stitcher = Stitcher(stitch_mode=stitch_mode, feature=feature, search_ratio=search_ratio, offset_match=offset_match)
    csv_file = pd.read_csv(input_csv)
    distance_loss = 0
    correct_num = 0
    time_start = time.time()
    false_image = []
    for idx in range(len(csv_file)):
        image_name = csv_file.iloc[idx, 1]
        print("the {} th images, Analysising {}".format(idx, image_name))
        local_start_time = time.time()
        imageA = cv2.imread(input_address + image_name + "\\" + image_name + "_A.jpg")
        imageB = cv2.imread(input_address + image_name + "\\" + image_name + "_B.jpg")
        drow = csv_file.iloc[idx, 2]
        dcol = csv_file.iloc[idx, 3]
        stitch_status, distance = stitcher.evaluateByFeatureSearch([imageA, imageB], [drow, dcol])
        distance_loss = distance_loss + distance
        if stitch_status:
            correct_num = correct_num + 1
        else:
            false_image.append(image_name)
        local_end_time = time.time()
        print('The duration time cost is {} s'.format(local_end_time - local_start_time))
        print('Now, the number of false match is {}'.format(len(false_image)))
    time_end = time.time()
    print('The duration time cost is {} s, and the average time cost is {}'.format(time_end - time_start, (time_end - time_start) / len(csv_file)))
    print('The average accuracy is {} %, and the average distance loss is {}'. format(correct_num / len(csv_file) * 100, distance_loss / len(csv_file)))
    # print("False Images: {}".format(false_image))
    f = open('tt.txt', 'w')
    f.write(str(false_image))
    f.close()