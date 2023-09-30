import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from single_vig import pvig_b_224_gelu
from models.PGAT import PGAT



class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dim=64):

        super(DynamicGraphConvolution, self).__init__()
        self.dim = dim

        self.qkv_dim=nn.Linear(dim,3*dim)

        self.relu = nn.LeakyReLU(0.2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Conv1d(in_features * 2, dim, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)





    def forward_construct_dynamic_graph(self, x):
        B, pe1, pe2= x.size()
        proj_qkv = self.qkv_dim(x)
        proj_qkv = proj_qkv.reshape(B,pe1, 3,pe2).permute(2,0,1,3)
        proj_query,proj_key,proj_value=proj_qkv[0],proj_qkv[1],proj_qkv[2]

        proj_key = proj_key.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)
        out = torch.bmm(attention, proj_value)

        x_glb = self.gamma * out + x
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.fc(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):  # 256.64.17

        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)  # 256.64.17

        return x

class DGNet(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes, patch_size, drop_prob=0.1, block_size=3):
        super(DGNet, self).__init__()
        self.input_channels = input_channels  # 32
        self.num_node = num_nodes  # 17
        self.num_classes = num_classes  # 17
        self.patch_size = patch_size  # 19



        # bone
        self.conv1 = nn.Conv2d(self.input_channels, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.gamma3 = nn.Parameter(torch.zeros(1))


        self.model1 = PGAT(img_sixe=11,size=7)
        self.model2 = PGAT(img_sixe=7,size=3)
        self.conv6 = nn.Conv2d(64, 64, 5)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 5)
        self.bn5 = nn.BatchNorm2d(64)

        self.relative = nn.Conv2d(64, 64, (1, 1), bias=False)
        self.transform = nn.Conv2d(64, 64, (1, 1))
        self.gcn = DynamicGraphConvolution(64, 64, dim=64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(4224, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.bn_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn_f2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def relation_compiler(self, x):  #  ¹ØÏµ±àÂëÆ÷Ä£¿é
        mask = self.relative(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)  #
        return x

    def forward(self, x):

        x = x.squeeze(1)

        x1 = F.leaky_relu(self.conv1(x))  # 256.32.17.17
        x1 = self.bn1(x1)

        x2 = F.leaky_relu(self.conv2(x1))  # 256.64.15.15
        x2 = self.bn2(x2)

        x3 = self.relation_compiler(x2)  # 256.64.17

        x4 = self.gcn(x3)

        x4 = self.gcn(x4) + x3
        x_4 = x4.view(-1, x4.size(1) * x4.size(2))  # 256.1088

        result = self.model1(x2)  # 256.64.11.11
        conv= F.leaky_relu(self.conv5(x2))
        conv = self.bn5(conv)
        result=result + conv

        x5 = self.model2(result)

        x6 = F.leaky_relu(self.conv6(result))
        x_6 = self.bn6(x6)

        x_6 = x_6 +x5

        x_7 = self.avgpool(x_6)
        x_8 = self.maxpool(x_6)
        x_7 = x_7.view(-1, x_7.size(1) * x_7.size(2) * x_7.size(3))
        x_8 = x_8.view(-1, x_8.size(1) * x_8.size(2) * x_8.size(3))

        x = torch.cat((x_7, x_8, self.gamma3 * x_4), dim=-1)

        x = self.fc1(x)

        x=self.drop1(x)
        x = self.bn_f1(x)
        x=F.leaky_relu(x)

        x = self.fc2(x)
        x = self.bn_f2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        return x,x

