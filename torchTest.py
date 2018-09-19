import torch
import torch.nn as nn

admp_channel = 32
feature_A = torch.randn(2, 3, 224, 224)
feature_B = feature_A.clone()

print((feature_A-feature_B).pow(2).sum())
print("feature_A")
print(feature_A)
admp32 = nn.AdaptiveMaxPool2d(admp_channel, return_indices=True)
batch_A, channel_A, high_A, width_A = feature_A.size()
batch_B, channel_B, high_B, width_B = feature_B.size()
response_A = feature_A.sum(1)
key, indices_A = admp32(response_A.unsqueeze(1))
indices_A = indices_A.view(batch_A, -1, 1)
drow_T = torch.ones(batch_A, admp_channel * admp_channel)
dcol_T = torch.ones(batch_A, admp_channel * admp_channel)
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
print("result")
print(drow_T.size())
print(drow_T.unique())
print(dcol_T.size())
print(dcol_T.unique())


# drow_T = torch.zeros(batch_A, admp_channel * admp_channel)
# dcol_T = torch.zeros(batch_A, admp_channel * admp_channel)
# for batch_index in range(batch_A):
#     response_A = feature_A[batch_index, :, :, :].sum(0)
#     _, indices_A = admp32(response_A.unsqueeze(0))
#     indices_A = indices_A.view(-1, 1)
#     for kp_index in range(indices_A.numel()):
#         row_A = indices_A[kp_index] / high_A
#         col_A = indices_A[kp_index] % high_A
#         descriptor_A = feature_A[batch_index, :, row_A, col_A]
#         distance = (feature_B[batch_index] - descriptor_A.unsqueeze(2).expand(-1, high_B, width_B)).pow(2).sum(0)
#         min_value = distance.min()
#         min_indice = distance.argmin()
#         row_B = min_indice / high_B
#         col_B = min_indice % high_B
#         drow = row_B - row_A
#         dcol = col_A - col_B
#         drow_T[batch_index, kp_index] = drow
#         dcol_T[batch_index, kp_index] = dcol
# print(drow_T.size())
# print(drow_T)
# print(dcol_T.size())
# print(dcol_T)

