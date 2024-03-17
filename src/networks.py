import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class AdaMSS_Seg(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.PET_encoder = Single_encoder(channel_num=8)
        self.CT_encoder = Single_encoder(channel_num=8)
        self.Fuse_encoder = Fuse_encoder(channel_num=16)
        self.Seg_decoder = Seg_decoder(channel_num=16)
        
    def forward(self, PET, CT):

        x_PET_1, x_PET_2, x_PET_3, x_PET_4, x_PET_5 = self.PET_encoder(PET)
        x_CT_1, x_CT_2, x_CT_3, x_CT_4, x_CT_5 = self.CT_encoder(CT)
        x_1, x_2, x_3, x_4, x_5 = self.Fuse_encoder(PET, CT,
                                                    x_PET_1, x_PET_2, x_PET_3, x_PET_4, x_PET_5,
                                                    x_CT_1, x_CT_2, x_CT_3, x_CT_4, x_CT_5)
        Seg_Tumor_pred, Seg_Node_pred = self.Seg_decoder(x_1, x_2, x_3, x_4, x_5)
        
        return [Seg_Tumor_pred, Seg_Node_pred]
    
    
class AdaMSS_Surv(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.PET_encoder = Single_encoder(channel_num=8)
        self.CT_encoder = Single_encoder(channel_num=8)
        self.Fuse_encoder = Fuse_encoder(channel_num=16)
        self.Surv_decoder = Surv_decoder(channel_num=16, interval_num=10)
        
    def forward(self, PET, CT, RadioClinic=None):

        x_PET_1, x_PET_2, x_PET_3, x_PET_4, x_PET_5 = self.PET_encoder(PET)
        x_CT_1, x_CT_2, x_CT_3, x_CT_4, x_CT_5 = self.CT_encoder(CT)
        x_1, x_2, x_3, x_4, x_5 = self.Fuse_encoder(PET, CT,
                                                    x_PET_1, x_PET_2, x_PET_3, x_PET_4, x_PET_5,
                                                    x_CT_1, x_CT_2, x_CT_3, x_CT_4, x_CT_5)
        Surv_pred, Regu_weight = self.Surv_decoder(x_1, x_2, x_3, x_4, x_5, RadioClinic=None)
        
        return [Surv_pred, Regu_weight]

    
#--------------------------------------------------------------------------------------     

class Single_encoder(nn.Module):

    def __init__(self, channel_num):
        super().__init__()
        
        self.RB_1 = Residual_block(1, channel_num, 2)
        self.RB_2 = Residual_block(channel_num, channel_num*2, 3)
        self.RB_3 = Residual_block(channel_num*2, channel_num*4, 3)
        self.RB_4 = Residual_block(channel_num*4, channel_num*8, 4)
        self.RB_5 = Residual_block(channel_num*8, channel_num*16, 4)
        
        self.MaxPool_1 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_2 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_3 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_4 = nn.MaxPool3d(2, stride=2)
        
    def forward(self, x_in):

        # full scale
        x_1 = self.RB_1(x_in)
        
        # downsample 1/2 scale
        x = self.MaxPool_1(x_1)
        x_2 = self.RB_2(x)
        
        # downsample 1/4 scale
        x = self.MaxPool_2(x_2)
        x_3 = self.RB_3(x)
        
        # downsample 1/8 scale
        x = self.MaxPool_3(x_3)
        x_4 = self.RB_4(x)
        
        # downsample 1/16 scale
        x = self.MaxPool_4(x_4)
        x_5 = self.RB_5(x)
        
        return x_1, x_2, x_3, x_4, x_5

    
class Fuse_encoder(nn.Module):

    def __init__(self, channel_num):
        super().__init__()
        
        self.Conv_1 = Conv_block(channel_num, channel_num, 3)
        self.Conv_2 = Conv_block(channel_num*2, channel_num*2, 3)
        self.Conv_3 = Conv_block(channel_num*4, channel_num*4, 3)
        self.Conv_4 = Conv_block(channel_num*8, channel_num*8, 3)
        self.Conv_5 = Conv_block(channel_num*16, channel_num*16, 3)
        
        self.RB_1 = Residual_block(2, channel_num, 2)
        self.RB_2 = Residual_block(channel_num, channel_num*2, 3)
        self.RB_3 = Residual_block(channel_num*2, channel_num*4, 3)
        self.RB_4 = Residual_block(channel_num*4, channel_num*8, 4)
        self.RB_5 = Residual_block(channel_num*8, channel_num*16, 4)
        
        self.MaxPool_1 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_2 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_3 = nn.MaxPool3d(2, stride=2)
        self.MaxPool_4 = nn.MaxPool3d(2, stride=2)
        
        self.FG_1 = Fusion_gate_block(channel_num)
        self.FG_2 = Fusion_gate_block(channel_num*2)
        self.FG_3 = Fusion_gate_block(channel_num*4)
        self.FG_4 = Fusion_gate_block(channel_num*8)
        self.FG_5 = Fusion_gate_block(channel_num*16)
        
    def forward(self, PET, CT,
               x_PET_1, x_PET_2, x_PET_3, x_PET_4, x_PET_5,
               x_CT_1, x_CT_2, x_CT_3, x_CT_4, x_CT_5):

        x = torch.cat([x_PET_1, x_CT_1], dim=1)
        x_con_1 = self.Conv_1(x)
        
        x = torch.cat([x_PET_2, x_CT_2], dim=1)
        x_con_2 = self.Conv_2(x)
        
        x = torch.cat([x_PET_3, x_CT_3], dim=1)
        x_con_3 = self.Conv_3(x)
        
        x = torch.cat([x_PET_4, x_CT_4], dim=1)
        x_con_4 = self.Conv_4(x)
        
        x = torch.cat([x_PET_5, x_CT_5], dim=1)
        x_con_5 = self.Conv_5(x)
        
        # Adaptive fuse encoder
        x = torch.cat([PET, CT], dim=1)
        x = self.RB_1(x)
        x_1 = self.FG_1(x, x_con_1)
        
        x = self.MaxPool_1(x_1)
        x = self.RB_2(x)
        x_2 = self.FG_2(x, x_con_2)
        
        x = self.MaxPool_2(x_2)
        x = self.RB_3(x)
        x_3 = self.FG_3(x, x_con_3)
        
        x = self.MaxPool_3(x_3)
        x = self.RB_4(x)
        x_4 = self.FG_4(x, x_con_4)
        
        x = self.MaxPool_4(x_4)
        x = self.RB_5(x)
        x_5 = self.FG_5(x, x_con_5)
        
        return x_1, x_2, x_3, x_4, x_5
    
    
class Seg_decoder(nn.Module):

    def __init__(self, channel_num):
        super().__init__()
        
        self.RB_6 = Residual_block(channel_num*16+channel_num*8, channel_num*8, 4)
        self.RB_7 = Residual_block(channel_num*8+channel_num*4, channel_num*4, 3)
        self.RB_8 = Residual_block(channel_num*4+channel_num*2, channel_num*2, 3)
        self.RB_9 = Residual_block(channel_num*2+channel_num, channel_num, 2)
        
        self.upsample_6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_9 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.Atten_gate_6 = Atten_gate_block(channel_num*8)
        self.Atten_gate_7 = Atten_gate_block(channel_num*4)
        self.Atten_gate_8 = Atten_gate_block(channel_num*2)
        self.Atten_gate_9 = Atten_gate_block(channel_num)
        
        self.Conv = nn.Conv3d(channel_num, 2, kernel_size=1, stride=1, padding='same')
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x_1, x_2, x_3, x_4, x_5):

        # upsample 1/8 scale
        x_gate = self.Atten_gate_6(x_4, x_5)
        x_up = self.upsample_6(x_5)
        x = torch.cat([x_gate, x_up], dim=1)
        x_6 = self.RB_6(x)
    
        # upsample 1/4 scale
        x_gate = self.Atten_gate_7(x_3, x_6)
        x_up = self.upsample_7(x_6)
        x = torch.cat([x_gate, x_up], dim=1)
        x_7 = self.RB_7(x)
        
        # upsample 1/2 scale
        x_gate = self.Atten_gate_8(x_2, x_7)
        x_up = self.upsample_8(x_7)
        x = torch.cat([x_gate, x_up], dim=1)
        x_8 = self.RB_8(x)
    
        # full scale
        x_gate = self.Atten_gate_9(x_1, x_8)
        x_up = self.upsample_9(x_8)
        x = torch.cat([x_gate, x_up], dim=1)
        x_9 = self.RB_9(x)
        
        # Segmentation output
        x = self.Conv(x_9)
        x = self.Sigmoid(x)
        Seg_Tumor_pred = x[:,0:1,:,:,:]
        Seg_Node_pred = x[:,1:2,:,:,:]
            
        return Seg_Tumor_pred, Seg_Node_pred
    
    
class Surv_decoder(nn.Module):

    def __init__(self, channel_num, interval_num, radioclinic_num=8, use_radioclinic=False):
        super().__init__()
        
        self.RB_6 = Residual_block(channel_num*16+channel_num*8, channel_num*8, 4)
        self.RB_7 = Residual_block(channel_num*8+channel_num*4, channel_num*4, 3)
        self.RB_8 = Residual_block(channel_num*4+channel_num*2, channel_num*2, 3)
        self.RB_9 = Residual_block(channel_num*2+channel_num, channel_num, 2)
        
        self.upsample_6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_9 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.Atten_gate_6 = Atten_gate_block(channel_num*8)
        self.Atten_gate_7 = Atten_gate_block(channel_num*4)
        self.Atten_gate_8 = Atten_gate_block(channel_num*2)
        self.Atten_gate_9 = Atten_gate_block(channel_num)
        
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

        if use_radioclinic:
            self.dense_1 = nn.Linear(channel_num*15+radioclinic_num, channel_num*8)
        else:
            self.dense_1 = nn.Linear(channel_num*15, channel_num*8)
        self.dense_2 = nn.Linear(channel_num*8, interval_num)
        
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x_1, x_2, x_3, x_4, x_5, RadioClinic=None):

        # upsample 1/8 scale
        x_gate = self.Atten_gate_6(x_4, x_5)
        x_up = self.upsample_6(x_5)
        x = torch.cat([x_gate, x_up], dim=1)
        x_6 = self.RB_6(x)
    
        # upsample 1/4 scale
        x_gate = self.Atten_gate_7(x_3, x_6)
        x_up = self.upsample_7(x_6)
        x = torch.cat([x_gate, x_up], dim=1)
        x_7 = self.RB_7(x)
        
        # upsample 1/2 scale
        x_gate = self.Atten_gate_8(x_2, x_7)
        x_up = self.upsample_8(x_7)
        x = torch.cat([x_gate, x_up], dim=1)
        x_8 = self.RB_8(x)
    
        # full scale
        x_gate = self.Atten_gate_9(x_1, x_8)
        x_up = self.upsample_9(x_8)
        x = torch.cat([x_gate, x_up], dim=1)
        x_9 = self.RB_9(x)
        
        # Survival output
        x_6 = torch.mean(x_6, dim=(2,3,4))
        x_7 = torch.mean(x_7, dim=(2,3,4))
        x_8 = torch.mean(x_8, dim=(2,3,4))
        x_9 = torch.mean(x_9, dim=(2,3,4))
        x = torch.cat([x_6, x_7, x_8, x_9], dim=1)
        
        # Clinical and radiomics features can be concatenated here
        if RadioClinic != None:
            x = torch.cat([x, RadioClinic], dim=1)
        
        x = self.dropout_1(x)
        x = self.dense_1(x)
        x = self.ReLU(x)
        
        x = self.dropout_2(x)
        x = self.dense_2(x)
        Surv_pred = self.Sigmoid(x)
        
        Regu_weight = [self.dense_1.weight, self.dense_2.weight]
        
        return Surv_pred, Regu_weight
    
    
#-------------------------------------------------------------------------------------- 

class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        
        self.Conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernels, stride=1, padding='same')
        self.BN = nn.BatchNorm3d(out_channels)
        self.ReLU = nn.ReLU()
        
    def forward(self, x_in):

        x = self.Conv(x_in)
        x = self.BN(x)
        x_out = self.ReLU(x)
        
        return x_out
    
    
class Residual_block(nn.Module):

    def __init__(self, in_channels, out_channels, conv_num):
        super().__init__()
        
        self.Conv_res = Conv_block(in_channels, out_channels, 1)
        self.Conv = Conv_block(in_channels, out_channels, 3)        
        
        self.Remain_Conv = nn.ModuleList()
        for i in range(conv_num-1):
            self.Remain_Conv.append(Conv_block(out_channels, out_channels, 3))
        
    def forward(self, x_in):

        x_res = self.Conv_res(x_in)
        x = self.Conv(x_in)
        x_out = torch.add(x, x_res)
        
        for Conv in self.Remain_Conv:
            x = Conv(x_out)
            x_out = torch.add(x, x_out)
        
        return x_out


class Fusion_gate_block(nn.Module):

    def __init__(self, channel_num):
        super().__init__()
        
        self.Conv_1 = nn.Conv3d(channel_num*2, channel_num, kernel_size=1, stride=1, padding='same')
        self.Conv_2 = nn.Conv3d(channel_num*2, channel_num, kernel_size=1, stride=1, padding='same')
        nn.init.zeros_(self.Conv_1.weight)
        nn.init.zeros_(self.Conv_2.weight)
        
    def forward(self, x_1, x_2):

        x = torch.cat([x_1, x_2], dim=1)
        w_1 = self.Conv_1(x)
        w_2 = self.Conv_2(x)
        
        # Softmax
        exp_1 = torch.exp(w_1)
        exp_2 = torch.exp(w_2)
        w_1 = exp_1/(exp_1+exp_2)
        w_2 = exp_2/(exp_1+exp_2)
        
        x_out = torch.add(torch.mul(x_1, w_1), torch.mul(x_2, w_2))
        return x_out


class Atten_gate_block(nn.Module):

    def __init__(self, channel_num):
        super().__init__()
        
        self.Conv_g = nn.Conv3d(channel_num*2, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g = nn.BatchNorm3d(channel_num)
        
        self.Conv_x = nn.Conv3d(channel_num, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x = nn.BatchNorm3d(channel_num)
        
        self.Conv_relu = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1, padding='same')
        self.BN_relu = nn.BatchNorm3d(1)
        
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        
        self.AvgPool = nn.AvgPool3d(2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x_in, g_in):

        g = self.Conv_g(g_in)
        g_int = self.BN_g(g)
        
        x = self.Conv_x(x_in)
        x = self.BN_x(x)
        x_int = self.AvgPool(x)
        
        x = torch.add(x_int, g_int)
        x_relu = self.ReLU(x)
        
        x = self.Conv_relu(x_relu)
        x = self.BN_relu(x)
        x = self.Sigmoid(x)
        x_mask = self.Upsample(x)
        
        x_out = torch.mul(x_in, x_mask)
        return x_out
