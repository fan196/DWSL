__all__ = ['backbone']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from layers.layers import *
from layers.RevIN import RevIN
from timm.models.layers import DropPath
from einops import rearrange

from .CGSLM import CGSL

class backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 n_layers: int = 3,d_model=128, dropout=0.0, pe: str = 'zeros', learn_pe: bool = True,
                 head_dropout=0, padding_patch=None, pretrain_head: bool = False, head_type='flatten',
                 individual=False, revin=True, affine=True, subtract_last=False,):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len,
                                    n_layers=n_layers,d_model=d_model,
                                    dropout=dropout, pe=pe, learn_pe=learn_pe)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z, train):
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        # model
        z = self.backbone(z,train)
        z = self.head(z)

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len,n_layers=3,d_model=128,  dropout=0.,
                 pe='zeros', learn_pe=True):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(c_in,q_len, d_model, dropout=dropout, n_layers=n_layers)

    def forward(self, x,train) -> Tensor:

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u,train)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)

        return z


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = torch.permute(x, (0, 2, 1))
        return x


class Reweight(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv1d(hidden_features, in_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWF(nn.Module):
    def __init__(self, q_len, dim, proj_drop=0.):
        super().__init__()
        self.fc_h = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.fc_h1 = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.theta_h_conv = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.Reweight = Reweight(q_len, 2 * q_len)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        theta_h = self.theta_h_conv(x)
        x_h = self.fc_h(x)
        x_h1 = self.fc_h1(x)
        x_h = self.linear(x_h * torch.cos(theta_h) + x_h1 * torch.sin(theta_h))
        Reweight = self.Reweight(F.adaptive_avg_pool1d(x_h, output_size=1)).softmax(dim=1)
        x_h = self.proj_drop(self.proj(x_h * Reweight))
        return x_h

class CGSLMM(nn.Module):
    def __init__(self, c_in, q_len, dim, mlp_ratio=4., drop_path=0.4, act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.c_in = c_in
        self.seg_num = q_len
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(q_len * mlp_ratio)
        self.mlp = Mlp(in_features=q_len, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.cgslm = CGSL(channels=dim,drop=drop_path)


    def forward(self, x, train):
        x = rearrange(x, '(b n) out_seg_num d_model -> (b out_seg_num) n d_model', n = self.c_in,out_seg_num=self.seg_num)
        x = self.norm1(x + self.drop_path(self.cgslm(x, train)))
        x = rearrange(x, '(b out_seg_num) n d_model -> (b n) out_seg_num d_model', n = self.c_in,out_seg_num=self.seg_num)
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class DWFM(nn.Module):
    def __init__(self, q_len, dim, mlp_ratio=4., drop_path=0.4, act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = DWF(q_len, dim, proj_drop=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(q_len * mlp_ratio)
        self.mlp = Mlp(in_features=q_len, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.attn(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, c_in, q_len, d_model, dropout=0., n_layers=1):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(DWFM(q_len, d_model, mlp_ratio=4., drop_path=dropout))
        self.layers1 = nn.ModuleList()
        for i in range(n_layers):
            self.layers1.append(CGSLMM(c_in, q_len, d_model, mlp_ratio=4., drop_path=dropout))

    def forward(self, src: Tensor, train:str):
        output = src
        i = 0
        for mod in self.layers:
            output = mod(output)
            output = self.layers1[i](output,train)
            i = i + 1
        return output




