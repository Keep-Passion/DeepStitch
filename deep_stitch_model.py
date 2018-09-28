import torch.nn as nn
import torch
import numpy as np
import torch.multiprocessing as mp


class DeepStitch(nn.Module):
    def __init__(self, feature_backbone, admp_channel, stitch_mode):
        super(DeepStitch, self).__init__()
        self.feature_backbone = feature_backbone
        self.admp_channel = admp_channel
        self.admp = nn.AdaptiveMaxPool2d(admp_channel, return_indices=True)
        self.stitch_mode = stitch_mode

    def featureSelect(self, feature):
        response_A = feature.sum(1)
        _, indices_A = self.admp(response_A.unsqueeze(1))
        return indices_A

    def getTranslationalOffsetByMode(self, array):
        zipped = zip(array[:, 0].tolist(), array[:, 1].tolist())
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        return dx, dy

    def caculateOffsetFromTwoFeature(self, tensors_shared, batch_index, kp_index):
        #     def caculateOffsetFromTwoFeature(self, q, indices_A, batch_index, kp_index, feature_A, feature_B):
        #     def caculateOffsetFromTwoFeature(self, q, indices_A, batch_index, kp_index, feature_A, feature_B, offset_T, min_value_T):
        #     def cacluateOffsetFromTwoFeature(self, batch_index, kp_index):
#         tensor_shared = [feature_A, feature_B, indices_A, offset_T, min_value_T]
        feature_A = tensors_shared[0]
        feature_B = tensors_shared[1]
        indices_A = tensors_shared[2]
        offset_T = tensors_shared[3]
        min_value_T = tensors_shared[4]
        batch_A, channel_A, high_A, width_A = feature_A.size()
        batch_B, channel_B, high_B, width_B = feature_B.size()
#         print("batch_index :{}".format(batch_index))
#         print("kp_index :{}".format(kp_index))
#         print("width_A :{}, width_B :{}".format(width_A, width_B))
#         print("indices_A size original :{}".format(indices_A.size()))
#         print("indices_A size :{}".format(tensors_shared[2][batch_index, kp_index, 0].size()))
        row_A = (indices_A[batch_index, kp_index] / width_A).item()
        col_A = (indices_A[batch_index, kp_index] % width_A).item()
        descriptor_A = feature_A[batch_index, :, row_A, col_A]
        distance = (feature_B[batch_index] - descriptor_A.unsqueeze(1).unsqueeze(2).expand(-1, high_B, width_B)).pow(2).sum(0)
        min_value = distance.min()
        min_indice = distance.argmin()
        row_B = (min_indice / width_B).item()
        col_B = (min_indice % width_B).item()
        # I don't know why, but that's right
        drow = -(row_B - row_A)
        dcol = -(col_B - col_A)
#         q.put([float(drow), float(dcol), float(min_value)])
        offset_T[batch_index, kp_index, 0] = float(drow)
        offset_T[batch_index, kp_index, 1] = float(dcol)
        min_value_T[batch_index, kp_index] = float(min_value)
        del distance, min_value, min_indice, descriptor_A, row_A, row_B, col_A, col_B, drow, dcol

    def forward(self, xA, xB, isTrainFeature=False, offset=None):
        if isTrainFeature:
            if self.stitch_mode == 0:
                feature_A = self.feature_backbone(xA)
                feature_B = self.feature_backbone(xB)
                batch_A, channel_A, high_A, width_A = feature_A.size()
                batch_B, channel_B, high_B, width_B = feature_B.size()
                # 训练特征模式
                drow = int(offset[0, 0].item())
                dcol = int(offset[0, 1].item())
                if drow >= 0 and dcol >= 0:
                    distance = (feature_A[:, :, drow:, dcol:] - feature_B[:, :, 0: high_B - drow,0: width_B - dcol]).pow(2).sum()
                elif drow > 0 and dcol < 0:
                    distance = (feature_A[:, :, drow:, 0: width_A + dcol] - feature_B[:, :, 0: high_B - drow,- dcol:]).pow(2).sum()
                elif drow < 0 and dcol > 0:
                    distance = (feature_A[:, :, 0: high_A + drow, dcol:] - feature_B[:, :, -drow:,0: width_B - dcol]).pow(2).sum()
                elif drow < 0 and dcol < 0:
                    distance = (feature_A[:, :, 0: high_A + drow:, 0: width_A + dcol] - feature_B[:, :, -drow:, -dcol:]).pow(2).sum()
                return distance
        else:
            feature_A = self.feature_backbone(xA)
            feature_B = self.feature_backbone(xB)
            batch_A, channel_A, high_A, width_A = feature_A.size()
            batch_B, channel_B, high_B, width_B = feature_B.size()
            indices_A = self.featureSelect(feature_A)
            indices_A = indices_A.view(batch_A, -1, 1)
            offset_T = torch.ones(batch_A, self.admp_channel * self.admp_channel, 2).to(indices_A.device)
            min_value_T = torch.ones(batch_A, self.admp_channel * self.admp_channel).to(indices_A.device)
            tensor_shared = [feature_A.to('cpu'), feature_B.to('cpu'), indices_A.to('cpu'), offset_T.to('cpu'), min_value_T.to('cpu')]
            for t in tensor_shared:
                t.share_memory_()
            processes = []
            num_processes = self.admp_channel * self.admp_channel * batch_A
            q = mp.Queue()
            for batch_index in range(batch_A):
                for kp_index in range(num_processes):
                    p = mp.Process(target=self.caculateOffsetFromTwoFeature,args=(tensor_shared, batch_index, kp_index))
                    # p = mp.Process(target=self.cacluateOffsetFromTwoFeature, args=(batch_index, kp_index))
                    p.daemon = True
                    p.start()
                    processes.append(p)
            for p in processes:
                p.join()
            offset_T = tensor_shared[3]
#             for batch_index in range(batch_A):
#                 for kp_index in range(self.admp_channel * self.admp_channel):
#                     tempList = q.get()
#                     offset_T[batch_index, kp_index, 0] = float(drow)
#                     offset_T[batch_index, kp_index, 1] = float(dcol)
#                     min_value_T[batch_index, kp_index] = float(min_value)
            #             torch.cuda.synchronize()
            #             torch.cuda.empty_cache()
            if self.stitch_mode == 0:  # For translational mode
                out = np.zeros((batch_A, 2), dtype=np.int)
                for batch_index in range(batch_A):
                    dx, dy = self.getTranslationalOffsetByMode(offset_T[batch_index, :, :].to("cpu").numpy().astype(int))
                    out[batch_index, 0] = dx
                    out[batch_index, 1] = dy
            elif self.stitch_mode == 1:  # For Homography mode
                _, min_value_indices = torch.sort(min_value_T)
                min_value_indices = min_value_indices[:, 0: locatedPointNum]
            del feature_A, feature_B, indices_A, offset_T, min_value_T
            return out


if __name__ == "__main__":
    filterDrow = nn.Sequential(
        nn.Linear(32 * 32, int(32 * 32 / 2)),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(int(32 * 32 / 2), 1),
    )
    drow_T = torch.ones(2, 32 * 32)
    out = filterDrow(drow_T)
