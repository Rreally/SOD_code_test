import torch
import torch.nn as nn
from Res2Net.res2net_v1b import res2net50_v1b
from dpt.models import DPTSegmentationModel

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # 初始化ViT作为ERP图像的编码器
        self.erp_encoder = model = DPTSegmentationModel(
            150,
            path=None,
            backbone="vitb_rn50_384",
        )
        
        # 使用预训练的Res2Net作为CMP图像的编码器
        self.cmp_encoder = res2net50_v1b(pretrained=True)
        
        # 调整输出通道数，以确保ERP和CMP特征在每个层次上的通道数一致
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1)  # 对Res2Net第一层特征
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1)  # 对Res2Net第二层特征
        self.conv1x1_3 = nn.Conv2d(1024, 512, kernel_size=1)  # 对Res2Net第三层特征
        self.conv1x1_4 = nn.Conv2d(2048, 512, kernel_size=1)  # 对Res2Net第四层特征

    def forward(self, x):
        # ERP图像特征提取
        erp_features = self.erp_encoder(x[0])  # 通过ViT提取ERP图像的全局特征
        # CMP图像特征提取
        cmp_features_1 = self.cmp_encoder(x[1])
        cmp_features_2 = self.cmp_encoder(x[2])
        cmp_features_3 = self.cmp_encoder(x[3])
        cmp_features_4 = self.cmp_encoder(x[4])
        cmp_features_5 = self.cmp_encoder(x[5])
        cmp_features_6 = self.cmp_encoder(x[6])

        cmp_features_1_0 = self.conv1x1_1(cmp_features_1[0])
        cmp_features_1_1 = self.conv1x1_2(cmp_features_1[1])
        cmp_features_1_2 = self.conv1x1_3(cmp_features_1[2])
        cmp_features_1_3 = self.conv1x1_4(cmp_features_1[3])

        cmp_features_2_0 = self.conv1x1_1(cmp_features_2[0])
        cmp_features_2_1 = self.conv1x1_2(cmp_features_2[1])
        cmp_features_2_2 = self.conv1x1_3(cmp_features_2[2])
        cmp_features_2_3 = self.conv1x1_4(cmp_features_2[3])

        cmp_features_3_0 = self.conv1x1_1(cmp_features_3[0])
        cmp_features_3_1 = self.conv1x1_2(cmp_features_3[1])
        cmp_features_3_2 = self.conv1x1_3(cmp_features_3[2])
        cmp_features_3_3 = self.conv1x1_4(cmp_features_3[3])

        cmp_features_4_0 = self.conv1x1_1(cmp_features_4[0])
        cmp_features_4_1 = self.conv1x1_2(cmp_features_4[1])
        cmp_features_4_2 = self.conv1x1_3(cmp_features_4[2])
        cmp_features_4_3 = self.conv1x1_4(cmp_features_4[3])

        cmp_features_5_0 = self.conv1x1_1(cmp_features_5[0])
        cmp_features_5_1 = self.conv1x1_2(cmp_features_5[1])
        cmp_features_5_2 = self.conv1x1_3(cmp_features_5[2])
        cmp_features_5_3 = self.conv1x1_4(cmp_features_5[3])

        cmp_features_6_0 = self.conv1x1_1(cmp_features_6[0])
        cmp_features_6_1 = self.conv1x1_2(cmp_features_6[1])
        cmp_features_6_2 = self.conv1x1_3(cmp_features_6[2])
        cmp_features_6_3 = self.conv1x1_4(cmp_features_6[3])

        cmp_layer_1 = [cmp_features_1_0, cmp_features_2_0, cmp_features_3_0, cmp_features_4_0, cmp_features_5_0, cmp_features_6_0]
        cmp_layer_2 = [cmp_features_1_1, cmp_features_2_1, cmp_features_3_1, cmp_features_4_1, cmp_features_5_1, cmp_features_6_1]
        cmp_layer_3 = [cmp_features_1_2, cmp_features_2_2, cmp_features_3_2, cmp_features_4_2, cmp_features_5_2, cmp_features_6_2]
        cmp_layer_4 = [cmp_features_1_3, cmp_features_2_3, cmp_features_3_3, cmp_features_4_3, cmp_features_5_3, cmp_features_6_3]
        cmp_features = [cmp_layer_1, cmp_layer_2, cmp_layer_3, cmp_layer_4]
        return erp_features, cmp_features
