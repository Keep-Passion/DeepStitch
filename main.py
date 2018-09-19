import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import shutil
from deep_stitch_model import DeepStitch
from datasets_Preprocess import StitchDataset
from vgg_nmp_backbone import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 2
learning_rate = 0.0001
admp_channel = 32
isTrainFeature = True
train_data_root = '.\\datasets\\graffiti\\images\\train\\'
val_csv = '.\\datasets\\graffiti\\fromCSV\\val_fixed224.csv'
val_data_root = '.\\datasets\\graffiti\\fromCSV\\images_fixed\\'


checkpoint_address = '.\\checkpoint\\Train_chp.pth.tar'
checkpoint_best_address = '.\\checkpoint\\model_best.pth.tar'
backbone = vgg13_nmp(pretrained=True)
data_transform = transforms.Compose([
    transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90), expand=True),
        transforms.RandomRotation((180, 180), expand=True),
        transforms.RandomRotation((270, 270), expand=True),
    ])
])
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_best_address)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        total_num = len(val_loader)
        correct_num = 0
        mseLoss = 0
        for i, sample in enumerate(val_loader):
            total_num = total_num + 1
            imageA = sample['imageA'].to(device)
            imageB = sample['imageB'].to(device)
            gt_offset = sample['offset'].to(device)
            if isTrainFeature:
                mseLoss = model(imageA, imageB)
            else:
                prd_offset = model(imageA, imageB, isTrainFeature=isTrainFeature, offset=gt_offset)
                mseLoss += criterion(prd_offset, gt_offset)
                if prd_offset.round() == gt_offset:
                    correct_num = correct_num + 1
            print('Validating: The average mse loss in validation set is {:.4f}, and the accuracy is {:.4f}'.format(mseLoss / total_num, correct_num / total_num))
    return mseLoss / total_num

def main():
    train_dataset = StitchDataset(isFromCSV=False, rootDir=train_data_root,
                                  isFixedLength=True, fixedLength=224,
                                  transform=data_transform, overlapRatio=[0.2, 0.8], cropRatio=[0.6, 0.9])
    val_dataset = StitchDataset(isFromCSV=True, csvFile=val_csv, rootDir=val_data_root)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    deepStitchModel = DeepStitch(feature_backbone=backbone).to(device)
    deepStitchModel = torch.nn.DataParallel(deepStitchModel).cuda()
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(deepStitchModel.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    last_val_loss = 1000000000
    for epoch in range(num_epochs):
        deepStitchModel.train()
        adjust_learning_rate(optimizer, epoch)
        for i, sample in enumerate(train_loader):
            # Move tensors to the configured device
            imageA = sample['imageA'].to(device)
            imageB = sample['imageB'].to(device)
            gt_offset = sample['offset'].to(device)
            # Forward pass
            if isTrainFeature:
                loss = deepStitchModel(imageA, imageB, isTrainFeature=isTrainFeature, offset=gt_offset)
            else:
                prd_offset = deepStitchModel(imageA, imageB)
                loss = criterion(gt_offset, prd_offset)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        val_loss = validate(val_loader, deepStitchModel, criterion)
        is_best = False
        if val_loss < last_val_loss:
            is_best = True
        save_checkpoint(deepStitchModel.state_dict(), is_best, filename=checkpoint_address)

if __name__ == "__main__":
    main()