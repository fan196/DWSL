
# Cell
from typing import  Optional
from torch import nn
from torch import Tensor


from layers.backbone import backbone


class Model(nn.Module):
    def __init__(self, configs, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten'):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        d_model = configs.d_model
        dropout = configs.dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.model = backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                              n_layers=n_layers, d_model=d_model, dropout=dropout,
                              pe=pe, learn_pe=learn_pe, head_dropout=head_dropout, padding_patch = padding_patch,
                              pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                              subtract_last=subtract_last)
    
    
    def forward(self, x,train):

        x = x.permute(0,2,1)
        x = self.model(x,train)
        x = x.permute(0,2,1)
        return x