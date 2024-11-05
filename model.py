import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.projection import C2EB
from fea_ext import FeatureExtractor

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(in_channels, out_channels, 1, stride=1))
#         self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)

#     def forward(self, x):
#         size = x.shape[2:]
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#         x1 = self.atrous_block1(x)
#         x2 = self.atrous_block6(x)
#         x3 = self.atrous_block12(x)
#         x4 = self.atrous_block18(x)
#         x = self.conv1(torch.cat([image_features,x1, x2, x3, x4], dim=1))

#         return x
    
class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv4 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outplanes),
                                             nn.ReLU(inplace=True))
        self.fusionconv = nn.Sequential(nn.Conv2d(outplanes * 5, outplanes, 1, bias=False),
                                        nn.BatchNorm2d(outplanes),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)
                                        )

    def forward(self, x):
        x1 = self.atrous_conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x4 = self.atrous_conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.fusionconv(x)

        return out
    

class SGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGM, self).__init__()
        self.cep = C2EB()
        # ASPP模块，用于多尺度特征提取
        self.aspp = ASPP(in_channels, out_channels)
        # 1x1卷积用于降维
        self.conv1x1 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)

    def forward(self, erp, s, cmp):
        cmp = self.cep(cmp)
        s = self.aspp(s)
        s = F.interpolate(s,erp.size()[2:], mode='bilinear', align_corners=False)
        # 将三种输入特征进行拼接
        x = torch.cat([erp, s, cmp], dim=1)
        x = self.conv1x1(x)
        return x
    

# class MFF(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MFF, self).__init__()
#         # 使用四个平行的空洞卷积，膨胀率分别为1, 2, 3, 4
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False)
#         self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # 对输入的每个特征层进行卷积，并求和融合
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x)
#         x4 = self.conv4(x)
#         fused_features = x1 + x2 + x3 + x4
#         return self.relu(fused_features)

class MFF(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(MFF, self).__init__()
        dilations = [1, 2, 3, 4]
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv4 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outplanes),
                                             nn.ReLU(inplace=True))
        self.fusionconv = nn.Sequential(nn.Conv2d(inplanes * 5, outplanes, 1, bias=False),
                                        nn.BatchNorm2d(outplanes),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)
                                        )

    def forward(self, x):
        x1 = self.atrous_conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x4 = self.atrous_conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.fusionconv(x)

        return out

    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

    
class CFM(nn.Module):
    def __init__(self, inplanes, is_upsample=False, is_downsample=False):
        super(CFM,self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(inplanes)
        self.is_upsample = is_upsample
        self.is_downsample = is_downsample
        self.mff = MFF(inplanes, inplanes)
        self.conv1 = nn.Conv2d(inplanes*2, inplanes, kernel_size=1)

    def forward(self, x1, x2, x3):
        if self.is_downsample:
            x1 = self.sa(nn.MaxPool2d(2)(x1))
        else:
            x1 = self.sa(x1)

        if self.is_upsample:
            x3 = self.sa(F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False))
        else:
            x3 = self.sa(x3)
        
        x2_1 = self.mff(x2)
        x2_2 = self.ca(x2_1)
        x1 = x1 * x2_1 * x2_2
        x3 = x3 * x2_1 * x2_2
        x = x1 + x3
        x = self.mff(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        return x
    
class SRB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SRB, self).__init__()
        self.out = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel // 2)
        self.conv2 = nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :self.out // 2, :, :], out2[:, self.out // 2:, :, :]

        return F.relu(w * out1 + b, inplace=True)
    
class FIAB(nn.Module):
    def __init__(self, in_channel, in_channel_a):
        super(FIAB, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_a, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)


    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

	#z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(down_2 * left, inplace=True)

        out = torch.cat((z2, z1, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)
    
class FARM(nn.Module):
    def __init__(self, inplanes, outplanes, inplanes_a):
        super(FARM, self).__init__()
        self.srb = SRB(256, outplanes)
        self.fiab = FIAB(inplanes, inplanes_a)
        
    def forward(self, a, sg, cf):
        x = self.fiab(sg, a, cf)
        x = self.srb(x)
        return x


class SCFANet(nn.Module):
    def __init__(self):
        super(SCFANet, self).__init__()
        self.feaext = FeatureExtractor()
        self.cep = C2EB()
        self.sgm1 = SGM(1024,128)
        self.sgm2 = SGM(1024,256)
        self.sgm3 = SGM(1024,512)
        self.sgm4 = SGM(1024,512)

        self.cfm1 = CFM(128, is_upsample=True)
        self.cfm2 = CFM(256, is_upsample=True, is_downsample=True)
        self.cfm3 = CFM(512, is_upsample=True, is_downsample=True)
        self.cfm4 = CFM(512, is_downsample=True)
        self.fam1 = FARM(128,128,128)
        self.fam2 = FARM(256,256,256)
        self.fam3 = FARM(512,512,256)
        self.srb = SRB(512,512)

        self.out_conv1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        erp_fea, cmp_fea = self.feaext(x)
        cmp_fea_4 = self.cep(cmp_fea[-1])
        s_fea = torch.cat([erp_fea[-1], cmp_fea_4], dim=1)

        sg_fea_1 = self.sgm1(erp_fea[0], s_fea, cmp_fea[0])
        sg_fea_2 = self.sgm2(erp_fea[1], s_fea, cmp_fea[1])
        sg_fea_3 = self.sgm3(erp_fea[2], s_fea, cmp_fea[2])
        sg_fea_4 = self.sgm4(erp_fea[3], s_fea, cmp_fea[3])

        cf_fea_1 = self.cfm1(sg_fea_1,sg_fea_1,sg_fea_2)
        cf_fea_2 = self.cfm2(sg_fea_1,sg_fea_2,sg_fea_3)
        cf_fea_3 = self.cfm3(sg_fea_2,sg_fea_3,sg_fea_4)
        cf_fea_4 = self.cfm4(sg_fea_3,sg_fea_4,sg_fea_4)

        a_fea_4 = self.srb(cf_fea_4)
        a_fea_3 = self.fam3(a_fea_4, sg_fea_3, cf_fea_3)
        a_fea_2 = self.fam2(a_fea_3, sg_fea_2, cf_fea_2)
        a_fea_1 = self.fam1(a_fea_2, sg_fea_1, cf_fea_1)

        st_1 = self.out_conv1(a_fea_1)
        st_2 = self.out_conv2(a_fea_2)
        st_3 = self.out_conv3(a_fea_3)
        st_4 = self.out_conv4(a_fea_4)

        # st_1 = F.interpolate(self.out_conv1(a_fea_1), scale_factor=4, mode='bilinear', align_corners=False)
        # st_2 = F.interpolate(self.out_conv2(a_fea_2), scale_factor=8, mode='bilinear', align_corners=False)
        # st_3 = F.interpolate(self.out_conv3(a_fea_3), scale_factor=16, mode='bilinear', align_corners=False)
        # st_4 = F.interpolate(self.out_conv4(a_fea_4), scale_factor=32, mode='bilinear', align_corners=False)
        return st_1, st_2, st_3, st_4