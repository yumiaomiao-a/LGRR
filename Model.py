import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np



class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff



class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)



class CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),
        )


        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.classifier =  torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 1, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2))


    def forward(self, x):	    	# x == [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]

        #pdb.set_trace()

        x = self.lastconv1(x_concat)    # x [128, 32, 32]
        x = self.lastconv2(x)    # x [64, 32, 32]
        x = self.lastconv3(x)    # x [1, 32, 32]

        # map_x = x.squeeze(1)
        map_x = x.view(-1, 32*32*1)
        map_x = self.classifier(map_x)

        return map_x#, x_concat, x_Block1, x_Block2, x_Block3, x_input


  

class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):
        super(CDCNpp, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

      

        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU())


        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')


        self.classifier =  torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 1, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2))


        
    def forward(self, x):	    	# x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)

        # pdb.set_trace()

        map_x = self.lastconv1(x_concat)

        map_x = map_x.squeeze(1)
        # map_x = map_x.view(-1, 32*32*1)
        # map_x = self.classifier(map_x)

        return map_x#, x_concat, attention1, attention2, attention3, x_input





class channel_wise_avgpool(nn.Module):
    def __init__(self, kernel_size=(1,2), stride=(1,2)):
        super(channel_wise_avgpool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, input):
        # 如维度为(3, 6, 4, 4) 交换为 (3, 4, 4, 6)
        print('input:',input.shape)
        input = input.transpose(3,1)
        print('input.transpose:',input.shape)
        input = F.avg_pool2d(input, self.kernel_size, self.stride)
        print("max_pool.",input.shape)
        input = input.transpose(3,1).contiguous()
        print("final_cross",input.shape)
        return input






######################################################################################
#define attention#################### 1th
class interactive_attention_1(nn.Module):
    def __init__(self):
        super(interactive_attention_1,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=(3, 3), padding=1,stride=(1, 1), bias=False),
                                   nn.Sigmoid())
        self.sSE_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
        self.sSE_11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
    def forward(self,f1,f2):
        # f1,f2 = torch.split(input,[16,16],dim=1)
        f11 = f1 * self.sSE_1(f1)
        f22 = f2 * self.sSE_11(f2)
        f_fusion = torch.cat((f11,f22),dim=1)
        f3 = self.conv1(f_fusion)
        f4 = self.conv2(f3)
        # print('%%++___________',f4.shape)
        f11,f22 = torch.split(f4,[1,1],dim=1)
        return f11,f22


#define attention#################### 1th
class interactive_attention_2(nn.Module):
    def __init__(self):
        super(interactive_attention_2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=(3, 3), padding=1,stride=(1, 1), bias=False),
                                   nn.Sigmoid())
        self.sSE_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
        self.sSE_11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
    def forward(self,f1,f2):
        # f1,f2 = torch.split(input,[16,16],dim=1)
        f11 = f1 * self.sSE_1(f1)
        f22 = f2 * self.sSE_11(f2)
        f_fusion = torch.cat((f11,f22),dim=1)
        f3 = self.conv1(f_fusion)
        f4 = self.conv2(f3)
        # print('%%++___________',f4.shape)
        f11,f22 = torch.split(f4,[1,1],dim=1)
        return f11,f22

    
#define attention#################### 1th
class interactive_attention_3(nn.Module):
    def __init__(self):
        super(interactive_attention_3,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=(3, 3), padding=1,stride=(1, 1), bias=False),
                                   nn.Sigmoid())
        self.sSE_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
        self.sSE_11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
    def forward(self,f1,f2):
        # f1,f2 = torch.split(input,[16,16],dim=1)
        f11 = f1 * self.sSE_1(f1)
        f22 = f2 * self.sSE_11(f2)
        f_fusion = torch.cat((f11,f22),dim=1)
        f3 = self.conv1(f_fusion)
        f4 = self.conv2(f3)
        # print('%%++___________',f4.shape)
        f11,f22 = torch.split(f4,[1,1],dim=1)
        return f11,f22


#define attention#################### 1th
class interactive_attention_4(nn.Module):
    def __init__(self):
        super(interactive_attention_4,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=(3, 3), padding=1,stride=(1, 1), bias=False),
                                   nn.Sigmoid())
        self.sSE_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
        self.sSE_11 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.Sigmoid())
    def forward(self,f1,f2):
        # f1,f2 = torch.split(input,[16,16],dim=1)
        f11 = f1 * self.sSE_1(f1)
        f22 = f2 * self.sSE_11(f2)
        f_fusion = torch.cat((f11,f22),dim=1)
        f3 = self.conv1(f_fusion)
        f4 = self.conv2(f3)
        # print('%%++___________',f4.shape)
        f11,f22 = torch.split(f4,[1,1],dim=1)
        return f11,f22


#split tensor
def window_partition(x,window_size):
    '''
    Args:
        param x: (B,H,W,C)
        param window_size(int): window size
    return:
        windows:(num_windows*B,window_size,window_size,C)
    '''
    x = x.permute(0,2,3,1)
    # print('+++++++++++',x.shape)
    B,H,W,C = x.shape
    x = x.view(B,H // window_size, window_size, W // window_size, window_size,C)
    # print('+++++++++++',x.shape)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(B,-1,1,(window_size*window_size*C))
    # print('&&&&&+++++++++++',windows.shape)
    return windows



class decoder(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')

        self.Block1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block4 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            basic_conv(64, 3, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(3),
            nn.ReLU())

    def forward(self, x):	    	# x [3, 256, 256]

        x = self.upsample(x)
        x = self.Block1(x)
        x = self.upsample(x)
        x = self.Block2(x)
        x = self.upsample(x)
        x = self.Block3(x)
        x = self.upsample(x)
        x4 = self.Block4(x)
        # print('++++==========',x4.shape)
        return x4

    
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class decoder_global(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(decoder_global, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.upsample16 = nn.Upsample(scale_factor=16,mode='nearest')

        self.Block1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Block4 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            basic_conv(64, 3, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.conv1x1_1 = conv1x1(128,128)
        self.conv1x1_2 = conv1x1(128,128)
        self.conv1x1_3 = conv1x1(128,128)
        self.conv1x1_4 = conv1x1(128,3)

    def forward(self, x):	    	# x [3, 256, 256]
        org = x

        x1 = self.upsample(x)
        x1 = self.Block1(x1)
        org1 = F.avg_pool2d(org, org.size()[2:])
        org1 = self.conv1x1_1(org1)
        org1 = F.interpolate(org1, size=x1.size()[2:],mode='bilinear', align_corners=True)
        # print('************8',org.shape,x1.shape,a1.shape)
        x1 = torch.mul(x1,org1)

        x2 = self.upsample(x1)
        x2 = self.Block2(x2)
        org2 = F.avg_pool2d(org, org.size()[2:])
        org2 = self.conv1x1_2(org2)
        org2 = F.interpolate(org2, size=x2.size()[2:],mode='bilinear', align_corners=True)
        x2 = torch.mul(x2,org2)

        x3 = self.upsample(x2)
        x3 = self.Block3(x3)
        org3 = F.avg_pool2d(org, org.size()[2:])
        org3 = self.conv1x1_3(org3)
        org3 = F.interpolate(org3, size=x3.size()[2:],mode='bilinear', align_corners=True)
        x3 = torch.mul(x3,org3)

        x4 = self.upsample(x3)
        x4 = self.Block4(x4)
        org4 = F.avg_pool2d(org, org.size()[2:])
        org4 = self.conv1x1_4(org4)
        org4 = F.interpolate(org4, size=x4.size()[2:],mode='bilinear', align_corners=True)
        x4 = torch.mul(x4,org4)
        # print('++++==========',x4.shape)
        return x4




decoder = decoder()
decoder_global = decoder_global()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out



def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1 || 0 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 224, 224]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 224, 224]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 224, 224]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 224, 224]
        return out



# FAD_Head = FAD_Head()


#######################################################
class CDCN_my(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7,image_shape=(3, 240, 240)):
        super(CDCN_my, self).__init__()
        self.shape = image_shape
        self.decoder = decoder
        self.decoder_global = decoder_global
        self.FAD_Head = FAD_Head(self.shape[-1])#生成frequency
        self.sam = SAM(3,64)
        self.sam0 = SAM(3,64)
      

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv1_new = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.Res_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.Res_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.Res_conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.Res_conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.Res_conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.avgpool = nn.AvgPool2d(240,240)

        self.conv2 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU() )

        self.conv11 = nn.Sequential(
            basic_conv(12, 3, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.sigmoid = nn.Sigmoid()

        self.conv22 = nn.Sequential(
            basic_conv(12, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU())


        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block4 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


        self.Block11 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,196,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            # basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(196,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block22 = nn.Sequential(
            # basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,196,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            # basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(196,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block33 = nn.Sequential(
            # basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,196,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            # basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(196,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.Block44 = nn.Sequential(
            # basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128,196,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            # basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(196,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


        self.lastconv1 = nn.Sequential(
            basic_conv(128*4, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.lastconv3 = nn.Sequential(
            basic_conv(128, 3, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Sigmoid())

        self.lastconv33 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.lastconv333 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.lastconv3333 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid())

        self.lastconv4 = nn.Sequential(
            basic_conv(128, 3, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Sigmoid())

        self.lastconv44 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.lastconv444 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.lastconv4444 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid())


        self.last = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))



        self.upsample = nn.Upsample(size=(15, 15), mode='bilinear',align_corners=True)


        self.AVP = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.AVP1 = nn.Sequential(nn.AdaptiveAvgPool2d(30),nn.Sigmoid())  #1

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gap_1 = nn.AdaptiveAvgPool2d(1)
        self.gap_3 = nn.AdaptiveAvgPool2d(3)
        self.gap_5 = nn.AdaptiveAvgPool2d(5)


        self.interactive_attention_1 = interactive_attention_1()
        self.interactive_attention_2 = interactive_attention_2()
        self.interactive_attention_3 = interactive_attention_3()
        self.interactive_attention_4 = interactive_attention_4()

        self.c1 = torch.nn.Sequential(
            torch.nn.Linear(256*8*8, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(300, 2))

        self.c11 = torch.nn.Sequential(
            torch.nn.Linear(3200, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(300, 2),
            torch.nn.Sigmoid())

        self.c22= torch.nn.Sequential(
            torch.nn.Linear(75, 2), 
            torch.nn.Sigmoid())


        self.c33 = torch.nn.Sequential(
            torch.nn.Linear(675, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2),
            torch.nn.Sigmoid())


        self.c44 = torch.nn.Sequential(
            torch.nn.Linear(675, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2))


        self.sigmoid = torch.nn.Sigmoid()

        self.concat = torch.nn.Sequential(
            torch.nn.Linear(224, 2))

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(224,224,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU())


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)



    def forward(self, x):	    	# x [3, 256, 256]

        ####################################original
        #residual　
        x_input = x

        ###### multi-scale residual module
        x = self.conv1_new(x) #240*240*64
        m1,m2,m3,m4 = torch.split(x,[16,16,16,16],dim=1)
        m11 = self.Res_conv1(m1)
        m22 = torch.add(m11,m2)
        m22 = self.Res_conv2(m22)
        m33 = torch.add(m22,m3)
        m33 = self.Res_conv3(m33)
        m44 = torch.add(m33,m4)
        m44 = self.Res_conv4(m44)
        m_all = torch.cat((m11,m22,m33,m44),dim=1)
        m_all = self.Res_conv5(m_all)
        x_ = m_all
        x_res = m_all-x



        ################### SAM ATTENTION
        x_org_f = x_input
        x_org_f = self.sam(x_org_f)
        # x_org_f = self.conv77(x_org_f)
        # x_org_f_att = self.conv111(x_org_f)
        #
        # x_org_f = x_org_f.mul(x_org_f_att)

        # x_org_f_att = channel_wise_avgpool(x_org_f)
        # print('---------',x_org_f_att.shape)




        x_res_1 = self.Block1(x_res)	    	    	# x [128, 120, 120]  以前是64channel
        # x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]
        x_f_1 = self.Block11(x_org_f)
        f11,f22 = self.interactive_attention_1(x_res_1,x_f_1)
        x_res_1 = x_res_1.mul(f11)
        x_f_1 = x_f_1.mul(f22)
        ## x_Block1_up = self.upsample(x_res_1)   # x [128, 32, 32]

        x_construct_120 = self.lastconv333(x_res_1)
        y_construct_120 = self.lastconv444(x_f_1)


        # x_res_60 = self.maxpool(x_res_1)
        # x_f_60 = self.maxpool(x_f_1)
        # x_res__30 = self.maxpool(x_res_60)
        # x_f__30 = self.maxpool(x_f_60)


        x_res_2 = self.Block2(x_res_1)	    	    	# x [128, 60, 60]
        # x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]
        x_f_2 = self.Block22(x_f_1)
        f33,f44 = self.interactive_attention_2(x_res_2,x_f_2)
        x_res_2 = x_res_2.mul(f33)
        x_f_2 = x_f_2.mul(f44)
        ## x_Block2_up = self.upsample(x_res_2)   # x [128, 32, 32]

        x_construct_60 = self.lastconv33(x_res_2)
        y_construct_60 = self.lastconv44(x_f_2)
        # x_construct_60_1 = x_construct_60[:,1:2,:,:]
        # x_res_2 = x_res_2.mul(x_construct_60_1)
        # x_f_2 = x_f_2.mul(x_construct_60_1)


        # x_res_2 = torch.add(x_res_2,x_res_60)
        # x_f_2 = torch.add(x_f_2,x_f_60)
        # x_res_30 = self.maxpool(x_res_2)
        # x_f_30 = self.maxpool(x_f_2)


        x_res_3 = self.Block3(x_res_2)	    	    	# x [128, 30, 30]
        # x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]
        x_f_3 = self.Block33(x_f_2)
        f55,f66 = self.interactive_attention_3(x_res_3,x_f_3)
        x_res_3 = x_res_3.mul(f55)
        x_f_3 = x_f_3.mul(f66)
        ## x_Block3_up = self.upsample(x_res_3)   # x [128, 32, 32]

        x_construct_30 = self.lastconv3(x_res_3)
        y_construct_30 = self.lastconv4(x_f_3)

        # x_res_33 = F.adaptive_avg_pool2d(x_res_3, (1, 1))
        # print('--------',x_res_33.shape)
        # x_res_3 = x_res_3.mul(x_res_33)
        # x_f_33 = F.adaptive_avg_pool2d(x_f_3, (1, 1))
        # x_f_3 = x_res_3.mul(x_f_33)



        # x_res_3 = torch.add(x_res_3,x_res_30)
        # x_res_3 = torch.add(x_res_3,x_res__30)
        # x_f_3 = torch.add(x_f_3,x_f_30)
        # x_f_3 = torch.add(x_f_3,x_f__30)


        # x_construct_240 = x_construct_240.permute(0,2,3,1)
        # x_construct_240 = x_construct_240[...,:1]
        # x_construct_240 = x_construct_240.permute(0,3,1,2)
        # x_res_3 = x_res_3.mul( x_construct_240)
        # x_f_3 = x_f_3.mul( x_construct_240)
        # print('---------------',x_construct_240.shape,x_f_3.shape)

        x_res_4 = self.Block4(x_res_3)	    	    	# x [128, 15, 15]
        # x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]
        x_f_4 = self.Block44(x_f_3)
        f77,f88 = self.interactive_attention_4(x_res_4,x_f_4)
        x_res_4 = x_res_4.mul(f77)
        x_f_4 = x_f_4.mul(f88)
        # x_Block4_up = self.upsample(x_res_4)   # x [128, 32, 32]

        x_construct_15 = self.lastconv3333(x_res_4)
        y_construct_15 = self.lastconv4444(x_f_4)

        # x_construct_15_1 = x_construct_15[:,1:2,:,:]
        # x_res_4 = x_res_4.mul(x_construct_15_1)
        # x_f_4 = x_f_4.mul(x_construct_15_1)

        # print('  ',x_construct_15.shape)
        # cc1 = x_construct_15.view(-1, 1*1*675)
        # cc2 = y_construct_15.view(-1, 1*1*675)
        # cc1 = self.c33(cc1)
        # cc2 = self.c44(cc2)

        x_res_4_consis = F.adaptive_avg_pool2d(x_res_4,(1,1))
        x_res_4_consis = x_res_4_consis.view(x_res_4_consis.size(0),-1)
        x_f_4_consis = F.adaptive_avg_pool2d(x_f_4,(1,1))
        x_f_4_consis = x_f_4_consis.view(x_f_4_consis.size(0),-1)
        x_consis = torch.cat((x_res_4_consis,x_f_4_consis), dim=1)




        # x_cls = x_construct_15.view(-1, 15*15*3)
        # y_cls = y_construct_15.view(-1, 15*15*3)
        # x_cls = self.c_cls(x_cls)
        # y_cls = self.c_cls(y_cls)

        # x_res_5 = x_res_4
        # x_res_5 = self.upsample30(x_res_5)
        # x_res_6 = self.Block3_cp(x_res_5)
        # # print('-------',x_res_6.shape,x_res_3.shape)
        # x_res_6 = torch.add(x_res_6,x_res_3)
        # x_res_7 = self.upsample60(x_res_6)
        # x_res_7 = self.Block2_cp(x_res_7)
        # x_res_8 = torch.add(x_res_7,x_res_2)
        # x_res_8 = self.upsample120(x_res_8)
        # x_res_8 = self.Block1_cp(x_res_8)
        # x_res_9 = torch.add(x_res_8,x_res_1)
        # x_res_9 = self.lastconv5(x_res_9)




        # print('----------',x_res_4.shape,x_f_4.shape)
        aa,bb,ccc,dd = x.shape
        w_res_c4 = window_partition(x_construct_15,5) #图像分块，窗口size=5
        w_f_c4 = window_partition(y_construct_15,5) #图像分块，窗口size=5
        # print('_________',w_res_c.shape)
        w_res_c4 = w_res_c4.view(aa,9,-1)[:,[4],:]
        w_f_c4 = w_f_c4.view(aa,9,-1)[:,[4],:]
        # print('————_________',w_res_c.shape)
        w_res_c4 = w_res_c4.view(aa,-1)
        w_f_c4 = w_f_c4.view(aa,-1)
        # print('%%%_________',w_res_c.shape)
        # # print('____________',x.shape,y.shape,x_res_4.shape,x_org_f_4.shape,x_construct.shape,w_res_c.shape)
        w_res_c4 = self.c22(w_res_c4)
        w_f_c4 = self.c22(w_f_c4)
        # w_res_c2 = self.sigmoid(w_res_c2)
        # w_f_c2 = self.sigmoid(w_f_c2)



        aa,bb,ccc,dd = x.shape
        w_res_c = window_partition(x_res_4,5) #图像分块，窗口size=5
        w_f_c = window_partition(x_f_4,5) #图像分块，窗口size=5
        # print('_________',w_res_c.shape)
        w_res_c = w_res_c.view(aa,9,-1)[:,[4],:]
        w_f_c = w_f_c.view(aa,9,-1)[:,[4],:]
        # print('————_________',w_res_c.shape)
        w_res_c = w_res_c.view(aa,-1)
        w_f_c = w_f_c.view(aa,-1)
        # print('%%%_________',w_res_c.shape)
        a,b = w_res_c.shape
        # # print('____________',x.shape,y.shape,x_res_4.shape,x_org_f_4.shape,x_construct.shape,w_res_c.shape)
        w_res_c = self.c11(w_res_c)
        w_f_c = self.c11(w_f_c)
        # w_res_c = self.sigmoid(w_res_c)
        # w_f_c = self.sigmoid(w_f_c)



        ######用中间的特征聚合做分类
        # # concat2 = torch.cat((x_res_2,x_f_2), dim=1)
        # # concat3 = torch.cat((x_res_3,x_f_3), dim=1)
        # # concat4 = torch.cat((x_res_4,x_f_4), dim=1)
        # up_concat2 = self.upsample(x_res_2)   # x [256, 15, 15]
        # up_concat3 = self.upsample(x_res_3)   # x [256, 15, 15]
        # up_concat4 = x_res_4   # x [256, 15, 15]
        # # concat_final = torch.add(up_concat2,up_concat3)
        # # concat_final = torch.add(concat_final,up_concat4)
        # concat_final = torch.cat((up_concat2,up_concat3,up_concat4), dim=1)
        # # concat_final = torch.cat((concat_final,up_concat4), dim=1)
        #
        # # print('－－//-----------', concat_final.shape)
        # concat_final = self.gap(concat_final)
        # # print('//-----------', concat_final.shape)
        #
        # concat_final = concat_final.view(-1, 1*1*384)
        # concat_c = self.concat(concat_final)




        #################################x_res_4池化成不同尺寸，串联判断
        # pool0 = x_res_4
        # pool1 = self.gap_1(pool0)
        # pool1 = self.conv1_1(pool1)
        # pool3 = self.gap_3(pool0)
        # pool3 = self.conv1_2(pool3)
        # pool5 = self.gap_5(pool0)
        # pool5 = self.conv1_3(pool5)
        # up1 = self.upsample(pool1)
        # up3 = self.upsample(pool3)
        # up5 = self.upsample(pool5)
        # concat = torch.cat((pool0,up1,up3,up5), dim=1)
        # concat = self.conv1_4(concat)
        # concat_final = self.gap(concat)
        # concat_final = concat_final.view(-1, 1*1*224)
        # concat_c = self.concat(concat_final)



        # concat = torch.cat((x_res_4,x_f_4), dim=1)
        # concat = concat.mul(x_construct_15[:,1:2,:,:])
        # concat = self.AVP(concat)
        # concat_c = concat.view(-1, 1*1*256)
        # concat_c = self.classi(concat_c)



        x_Block1 = x_res_1
        x_Block2 = x_res_2
        x_Block3 = x_res_3
        x_Block4 = x_res_4

        x_Block11 = x_f_1
        x_Block22 = x_f_2
        x_Block33 = x_f_3
        x_Block44 = x_f_4

       
 
        return x_consis,x_,w_res_c,w_f_c,w_res_c4,w_f_c4,x_res_4,x_f_4,x_construct_30,x_construct_120,x_construct_60,x_construct_15,y_construct_30,y_construct_120,y_construct_60,y_construct_15,x_res,x_Block1,x_Block2,x_Block3,x_Block4,x_org_f,x_Block11,x_Block22,x_Block33,x_Block44


CDCNpp = CDCNpp()
CDCN_my = CDCN_my()

