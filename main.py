%matplotlib inline
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import shutil
from deep_stitch_model import DeepStitch
from datasets_preprocess import *
from vgg_nmp_backbone import *
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 1
learning_rate = 0.0001
admp_channel = 32
isTrainFeature = False
isAddExistModel = False
stitch_mode = 0
train_data_root = './datasets/graffiti/images/train/'
val_data_root = './datasets/graffiti/images/val/'

# val_csv = './datasets/graffiti/fromCSV/val_fixed224.csv'
# val_rootDir_csv = './datasets/graffiti/fromCSV/images_fixed/'

val_csv = './datasets/MSCOCO2014/fromCSV/val_notfixed.csv'
val_rootDir_csv = './datasets/MSCOCO2014/fromCSV/images_notfixed/'

checkpoint_address = './checkpoint/Train_chp.pth.tar'
checkpoint_best_address = './checkpoint/model_best.pth.tar'
backbone = vgg11_bn_nmp(pretrained=True)
both_trsf = transforms.Compose([
    transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90), expand=True),
        transforms.RandomRotation((180, 180), expand=True),
        transforms.RandomRotation((270, 270), expand=True),
    ])
])
second_trsf = transforms.Compose([transforms.ColorJitter()])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_best_address)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def stitchImagePairs(showResult = True):
    isTrainFeature = False
    val_dataset = StitchDataset(isFromCSV=True, csvFile=val_csv, rootDir=val_rootDir_csv)
#     val_dataset = StitchDataset(isFromCSV=False, rootDir=val_data_root, isFixedLength=False,
#                                 overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    model = DeepStitch(feature_backbone=backbone, admp_channel=admp_channel, stitch_mode=stitch_mode).to(device)
    if isAddExistModel:
        model.load_state_dict(torch.load(checkpoint_best_address))
    model.eval()
    time_start = time.time()
    false_image = []
    with torch.no_grad():
        total_num = len(val_dataset)
        correct_num = 0
        mse_loss = 0
        for i, sample in enumerate(val_loader):
            local_start_time = time.time()
            imageA = sample['imageA'].to(device)
            imageB = sample['imageB'].to(device)
            gt_offset = sample['offset'].squeeze(2).numpy().astype(int)
            image_name = sample['image_name']
            print("The " + str(i) + " th images. Analysising " + str(image_name))
            batch_num = imageA.size()[0]
            prd_offset = model(imageA, imageB, isTrainFeature=False)
            mse_loss = mse_loss + np.linalg.norm(prd_offset - gt_offset)
            if stitch_mode == 0:
                for batch_index in range(batch_num):
                    if (prd_offset[batch_index, :] == gt_offset[batch_index, :]).all():
                        correct_num = correct_num + 1
                    else:
                        false_image.append(image_name)
                    if showResult:
                        temp_imageA = imageA[batch_index, :, :, :].to('cpu').numpy().transpose().astype(np.uint8)
                        temp_imageB = imageB[batch_index, :, :, :].to('cpu').numpy().transpose().astype(np.uint8)
                        plt.subplot(131)
                        plt.imshow(temp_imageA)
                        plt.title("ImageA")
                        plt.subplot(132)
                        plt.imshow(temp_imageB)
                        plt.title("ImageB")
                        plt.subplot(133)
                        plt.imshow(getStitchByOffset([temp_imageA, temp_imageB], [prd_offset[batch_index, 1].item(), prd_offset[batch_index, 0].item()]))
                        plt.title("Stitched Result")
                        plt.show()
                    print("gt_offset = {}, prd_offset = {}, the result is {}".format(gt_offset[batch_index, :], prd_offset[batch_index, :], (prd_offset[batch_index, :] == gt_offset[batch_index, :]).all()))
            elif stitch_mode == 1:
                pass
            local_end_time = time.time()
            print('The duration time cost is {} s'.format(local_end_time - local_start_time))
            print('Now, the number of false match is {}, and they are: {}'.format(len(false_image), false_image))
        time_end = time.time()
        if stitch_mode == 0:
            print('The duration time cost is {} s, and the average time cost is {}'.format(time_end - time_start, (time_end - time_start) / total_num))
            print('Validating: The average mse loss in validation set is {:.4f}, and the accuracy is {:.4f}'.format(mse_loss / total_num, correct_num / total_num))
            print("False Images: {}".format(false_image))

def valiateFeature(val_dataset, model):
    isTrainFeature = True
    model.eval()
    mse_loss = 0
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    with torch.no_grad():
        total_num = len(val_dataset)
        for i, sample in enumerate(val_loader):
            # Move tensors to the configured device
            imageA = sample['imageA'].to(device)
            imageB = sample['imageB'].to(device)
            gt_offset = sample['offset'].to(device).squeeze(2)
            loss = model(imageA, imageB, isTrainFeature=isTrainFeature, offset=gt_offset, stitch_mode = stitch_mode)
            mse_loss = mse_loss + float(loss)
        print('Validating: mse_loss of validating dataset is : {:.4f}'
              .format(mse_loss / total_num))
    return mse_loss / total_num

def trainFeature():
    isTrainFeature = True
    train_dataset = StitchDataset(isFromCSV=False, rootDir=train_data_root,
                                  isFixedLength=True, fixedLength=224,
                                  both_trsf = both_trsf, second_trsf = second_trsf, overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_dataset = StitchDataset(isFromCSV=True, csvFile=val_csv, rootDir=val_data_root)

    deepStitchModel = DeepStitch(feature_backbone=backbone, admp_channel=admp_channel, stitch_mode=stitch_mode).to(device)
    if isAddExistModel:
        deepStitchModel.load_state_dict(torch.load(checkpoint_best_address))
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(deepStitchModel.parameters(), lr=learning_rate, momentum=0.9)
    total_step = len(train_loader)
    last_val_loss = 1000000000
    for epoch in range(num_epochs):
        deepStitchModel.train()
        adjust_learning_rate(optimizer, epoch)
        for i, sample in enumerate(train_loader):
            # Move tensors to the configured device
            imageA = sample['imageA'].to(device)
            imageB = sample['imageB'].to(device)
            gt_offset = sample['offset'].to(device).squeeze(2)
            loss = deepStitchModel(imageA, imageB, isTrainFeature=isTrainFeature, offset=gt_offset)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        val_loss = valiateFeature(val_dataset, deepStitchModel)
        is_best = False
        if val_loss < last_val_loss:
            is_best = True
            last_val_loss = val_loss
        save_checkpoint(deepStitchModel.state_dict(), is_best, filename=checkpoint_address)

if __name__ == "__main__":
    stitchImagePairs(False)
#     trainFeature()