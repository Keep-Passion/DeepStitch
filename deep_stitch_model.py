import torch.nn as nn
import torch

class DeepStitch(nn.Module):
    def __init__(self, feature_backbone, admp_channel):
        super(DeepStitch, self).__init__()
        self.feature_backbone = feature_backbone
        self.admp_channel = admp_channel
        self.admp = nn.AdaptiveMaxPool2d(admp_channel, return_indices=True)
        self.filterDrow = nn.Sequential(
            nn.Linear(int(self.admp_channel * self.admp_channel), int(self.admp_channel * self.admp_channel / 2)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(self.admp_channel * self.admp_channel / 2), 1),
        )
        self.filterDcol = nn.Sequential(
            nn.Linear(int(self.admp_channel * self.admp_channel), int(self.admp_channel * self.admp_channel / 2)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(self.admp_channel * self.admp_channel / 2), 1),
        )

    def featureSelect(self, feature):
        response_A = feature.sum(1)
        _, indices_A = self.admp(response_A.unsqueeze(1))
        return indices_A

    def forward(self, xA, xB, isTrainFeature=False, offset=None):
        if isTrainFeature:
            feature_A = self.feature_backbone(xA)
            feature_B = self.feature_backbone(xB)
            batch_A, channel_A, high_A, width_A = feature_A.size()
            batch_B, channel_B, high_B, width_B = feature_B.size()
            # 训练特征模式
            drow = int(offset[0, 0].item())
            dcol = int(offset[0, 1].item())
            if drow >= 0 and dcol >= 0:
                distance = (feature_A[:, :, drow:, dcol:] - feature_B[:, :, 0: high_B - drow, 0: width_B - dcol]).pow(2).sum()
            elif drow > 0 and dcol < 0:
                distance = (feature_A[:, :, drow:, 0: width_A + dcol] - feature_B[:, :, 0: high_B - drow, -dcol:]).pow(2).sum()
            elif drow < 0 and dcol > 0:
                distance = (feature_A[:, :, 0: high_A + drow, dcol:] - feature_B[:, :, -drow:, 0: width_B - dcol]).pow(2).sum()
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
            drow_T = torch.ones(batch_A, self.admp_channel * self.admp_channel).to(indices_A.device)
            dcol_T = torch.ones(batch_A, self.admp_channel * self.admp_channel).to(indices_A.device)
            for batch_index in range(batch_A):
                for kp_index in range(indices_A[batch_index].numel()):
                    row_A = indices_A[batch_index, kp_index] / high_A
                    col_A = indices_A[batch_index, kp_index] % high_A
                    descriptor_A = feature_A[batch_index, :, row_A, col_A]
                    distance = (feature_B[batch_index] - descriptor_A.unsqueeze(2).expand(-1, high_B, width_B)).pow(2).sum(0)
                    min_value = distance.min()
                    min_indice = distance.argmin()
                    row_B = min_indice / high_B
                    col_B = min_indice % high_B
                    drow = row_B - row_A
                    dcol = col_A - col_B
                    drow_T[batch_index, kp_index] = drow
                    dcol_T[batch_index, kp_index] = dcol
            out_drow = self.filterDrow(drow_T)
            out_dcol = self.filterDcol(dcol_T)
            out = torch.cat([out_drow, out_dcol], 1)
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
