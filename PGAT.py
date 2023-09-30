# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


import numpy as np

import torch

from torch.nn import Sequential as Seq, Linear as Lin, Conv2d




def get_2d_relative_pos_embed(embed_dim, grid_size):

    #embed_dim=32
    #grid_size=19
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)#361.32
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]#361.361
    return relative_pos

# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size, dtype=np.float32)#0到18个数字
    grid_w = np.arange(grid_size, dtype=np.float32)#0到18个数字
    grid = np.meshgrid(grid_w, grid_h)# 生成网格grid点坐标矩阵 # here w goes first 这里 w 先行
    grid = np.stack(grid, axis=0)#2.19.19这个函数的作用就是堆叠作用，就是将两个分离的数据堆叠到一个数据里

    grid = grid.reshape([2, 1, grid_size, grid_size])#2.1.19.19
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)#361.32
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)#np.concatenate()是用来对数列或矩阵进行合并的
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])#361.16  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])#361.16  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)#361.32 # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)#0到7 8个数
    omega /= embed_dim / 2.#数组里面的数字都除以8
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega) #361.8 # (M, D/2), outer product 外积

    emb_sin = np.sin(out) # (M, D/2)#361.8
    emb_cos = np.cos(out) # (M, D/2)#361.8

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)#361.16
    return emb


def pairwise_distance(x):#2.361.32

    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))#2.361.361
        #对两个张量进行逐元素乘法
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)#2.361.1
        return x_square + x_inner + x_square.transpose(2, 1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        expanded_weight = self.W.unsqueeze(0)
        Wh = torch.matmul(h, expanded_weight)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)#196*196
        h_prime = torch.matmul(attention, Wh)#196*30

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        M = Wh.size()[0]
        N = Wh.size()[1] #196 number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1,N, 1)#38416*30
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(M,N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nout, dropout, nheads,alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.squeeze(-1).permute(0, 2, 1)
        # stacked_tensors = torch.stack([att(x, adj) for att in self.attentions], dim=1)
        # x = torch.sum(stacked_tensors, dim=1)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)#196*120
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))#196*64
        return x

def matrix(x, k=16, relative_pos=None):

    #x=2.32.361.1    relative_pos=1.361.361
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)#2.361.32
        batch_size, n_points, n_dims = x.shape



        dist = pairwise_distance(x.detach())#2.361.361
        if relative_pos is not None:
            dist += relative_pos#256.121.121
        _, nn_idx = torch.topk(-dist, k=k)

        zero_vec = torch.zeros_like(dist)
        adj = zero_vec.scatter(-1, nn_idx, 1)
        # attention = torch.where(adj > 0, dist, adj)
        # attention = F.softmax(attention, dim=2)

    return adj


class lin(nn.Module):


    def __init__(self, nfeat, nhid, nout, dropout, nheads,alpha):
        """Dense version of GAT."""
        super(lin, self).__init__()
        self.GAT_Branch = GAT(nfeat, nhid, nout, dropout, nheads, alpha)


    def forward(self, x, adj):
        x = self.GAT_Branch(x, adj)
        return x


class Graph(nn.Module):

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Graph, self).__init__()
        self.dilation = dilation   #1
        self.stochastic = stochastic  #Flase
        self.epsilon = epsilon  #0.2
        self.k = k   #9
        self.lin1=lin(nfeat=64, nhid=15, nout=64, dropout=0.4, nheads=4,alpha=0.2)

    def forward(self, x, y=None, relative_pos=None):

        x = F.normalize(x, p=2.0, dim=1)#将某一个维度除以那个维度对应的范数(默认是2范数)。2.32.361.1表示得是361是H和W相乘
        adj = matrix(x, self.k * self.dilation, relative_pos)#2.2.361.9
        #return self._dilated(edge_index)

        out=self.lin1(x,adj)

        return out



class GraphGAT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(GraphGAT, self).__init__()
        self.k = kernel_size    #9
        self.d = dilation      #1
        self.r = r        #1
        self.GGAT = Graph(kernel_size, dilation, stochastic, epsilon)#密集扩张 Knn 图

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape#2.32.19.19
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()#2.32.361.1
        x = self.GGAT(x, y, relative_pos)#2.2.361.9

        return x.permute(0,2,1).reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels #32
        self.n = n   #HW 19*19
        self.r = r   #1
        self.layer1 = nn.LeakyReLU()

        #一层图卷积
        self.PoGAT = GraphGAT(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        #在一层卷积层
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        #输入是啥，直接给输出，不做任何的改变 nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None  #使用相对位置
        if relative_pos:
            print('using relative_pos')

            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)

            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            #liu=2   debug之后进来了
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x  #2.32.19.19
        B, C, H, W = x.shape  #2.32.19.19
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)#1.361.361
        #x=2.32.19.19    relative_pos=1.361.361
        x = self.PoGAT(x, relative_pos)#2.64.19.19
        x=self.layer1(x)
        x = self.Conv(x)#2.32.19.19
        x = self.drop_path(x) + _tmp#2.32.19.19

        return x




class DeepGAT(torch.nn.Module):
    def __init__(self,opt):
        super(DeepGAT, self).__init__()
        print(opt)
        k = opt.k  #9
        act = opt.act  #gelu
        norm = opt.norm  #batch
        bias = opt.bias   #true
        epsilon = opt.epsilon   #0.2gcn的随机ε
        stochastic = opt.use_stochastic#flase
        conv = opt.conv#mr

        self.size=opt.size

        blocks = opt.blocks#[2.2.18.2]
        self.n_blocks = sum(blocks)#21
        channels = opt.channels#[32.128.320.512]
        self.img_size = opt.img_size#19

        HW = self.img_size * self.img_size  #361


        self.backbone_1 = nn.ModuleList([])
        self.number = [2]
        for j in range(self.number[0]):
            self.backbone_1 += [
                Seq(Grapher(channels[0], k, 1, conv, act, norm,
                            bias, stochastic, epsilon, 1, n=HW, drop_path=0,
                            relative_pos=True))]
        self.backbone_1 = Seq(*self.backbone_1)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)#使用正态分布对输入张量进行赋值
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        for i in range(len(self.backbone_1)):
            x = self.backbone_1[i](inputs)

        B,C,H,W=x.shape
        x = x.view(B, C, -1)
        x = x[:, :, math.ceil((H * W) / 2) - math.ceil((self.size ** 2)/2):(math.ceil((H * W) / 2)) +math.floor((self.size ** 2)/2)]

        x = x.view(B, C, self.size,self.size)

        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }



def PGAT(**kwargs):
    class OptInit:
        def __init__(self, num_classes=2, drop_path_rate=0.0,img_sixe=7, size=7,**kwargs):
            self.k = 12 # neighbor num (default:9)
            self.conv = 'edge'  # graph conv layer {edge, mr,sage,gin}
            self.act = 'leakyrelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}激活层
            self.norm = 'batch'  # batch or instance normalization {batch, instance}批量或实例规范化
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn  gcn的随机ε
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2]  # number of basic blocks in the backbone主干中的基本块数
            self.channels = [64, 128]  # number of channels of deep features[48, 96, 240, 384][128, 256, 512, 1024]深度特征的通道数
            self.n_classes = num_classes  # Dimension of out_channelsout_channels 的维度
            self.emb_dims = 1024  # Dimension of embeddings 嵌入的维度
            self.img_size = img_sixe
            self.size = size


    opt = OptInit(**kwargs)
    model = DeepGAT(opt)
    return model
