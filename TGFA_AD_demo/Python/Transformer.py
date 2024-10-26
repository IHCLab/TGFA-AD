import torch
from torch import nn
from torchmetrics import TotalVariation
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Tokenization(nn.Module):
    def __init__(self,patch_size):
        super(Tokenization,self).__init__()
        self.rearrange = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1]),)  
    def forward(self,x):
        x=self.rearrange(x).permute(0,2,1)
        return  x

class CrossAttention(nn.Module):
    def __init__(self, endmember,num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1,cls_token=False,numpatch=2500):
        super(CrossAttention,self).__init__()
        self.cls_token=cls_token
        self.num_heads = num_heads
        head_dim = endmember // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(endmember, endmember, bias=qkv_bias)
        self.wk = nn.Linear(endmember, endmember, bias=qkv_bias)
        self.wv = nn.Linear(endmember, endmember, bias=qkv_bias)
        self.proj = nn.Linear(endmember, endmember)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.cls_token:
            q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = self.attn_drop(attn)
        if self.cls_token:
            x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        else:
            x = ( attn @ v).transpose(1, 2).reshape(B, N, C)  
        x = self.proj(x)
        x = self.proj_drop(x)
        return x    
    
class LayerNorm(nn.Module):
    def __init__(self, endmember, fn):
        super(LayerNorm,self).__init__()
        self.norm = nn.LayerNorm(endmember)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, endmember, hidden_dim, dropout=0.1):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(endmember, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, endmember),
            nn.Sigmoid(),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class T_Former(nn.Module):
     def __init__(self, endmember, hidden_dim, num_patches,patch_pixel, dropout=0.1):
        super(T_Former,self).__init__()
        self.norm=nn.LayerNorm(endmember)
        self.layers = nn.ModuleList([])
        self.local_pos= nn.Parameter(torch.randn(num_patches, patch_pixel + 1, endmember))
        self.cls= nn.Parameter(torch.randn(num_patches, 1, endmember)) 
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                (LayerNorm(endmember, CrossAttention(endmember,cls_token=True))),
                Residual(LayerNorm(endmember, FeedForward(endmember, hidden_dim, dropout=dropout)))
            ]))
     def forward(self,x):
        x_hat=self.norm(torch.cat([self.cls,x],dim=1)+self.local_pos)
        for attn, ff in self.layers:
            x_hat=attn(x_hat)
            x_hat=ff(x_hat)
            x_hat=torch.cat([x_hat,x],dim=1)
        return x_hat[:, 0:1, ...]
    
class C_Former(nn.Module):
    def __init__(self,endmember,num_patches, dropout=0.1):
        super(C_Former,self).__init__()
        self.norm=nn.LayerNorm(num_patches)
        self.layers = nn.ModuleList([])
        self.channel_pos=nn.Parameter(torch.randn(1, endmember,num_patches))
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                Residual(LayerNorm(num_patches, CrossAttention(num_patches,num_heads=10))),
                Residual(LayerNorm(num_patches, FeedForward(num_patches, num_patches//2, dropout=dropout))) ]))
    def forward(self,x):
        x_hat=self.norm(x+self.channel_pos)
        x_hat=self.norm(x)
        for attn, ff in self.layers:
            x_hat=attn(x_hat)
            x_hat=ff(x_hat)
        return x_hat

class S_Former(nn.Module):
    def __init__(self,endmember,num_patches, dropout=0.1):
        super(S_Former,self).__init__()
        self.norm=nn.LayerNorm(endmember)
        self.layers = nn.ModuleList([])
        self.channel_pos=nn.Parameter(torch.randn(1, num_patches,endmember))
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                Residual(LayerNorm(endmember, CrossAttention(endmember))),
                Residual(LayerNorm(endmember, FeedForward(endmember, 2, dropout=dropout)))]))
    def forward(self,x):
        x_hat=self.norm(x+self.channel_pos)
        x_hat=self.norm(x)
        for attn, ff in self.layers:
            x_hat=attn(x_hat)
            x_hat=ff(x_hat)
        return x_hat
    
class ResCNN(nn.Module):
    def __init__(self,endmember):
        super(ResCNN,self).__init__()
        self.layers = nn.ModuleList([])
        self.conv1=nn.Conv2d(endmember,endmember,1,padding=0)
        for _ in range(2):
            self.layers.append(nn.ModuleList([
                nn.Conv2d(endmember,endmember,3,padding=1),
                nn.Sigmoid(), ]))
    def forward(self,x):
        z=x
        for conv3,re in self.layers:
            x=re(conv3(x))
        x=self.conv1(x)+z
        return x 
    
class Decoder(nn.Module):
     def __init__(self,input_size,num_patches):
        super(Decoder,self).__init__()
        self.image_size=input_size
        pixel_num=input_size[2]*input_size[3]
        self.layer=nn.Sequential(
        nn.Linear(num_patches,pixel_num//4),
        nn.Sigmoid(),
        nn.Linear(pixel_num//4,pixel_num),)
        
     def _init_weights(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.normal_(self.layer.bias, std=1e-6)

     def forward(self,x):
         x=self.layer(x).squeeze()
         x=x.reshape(-1,self.image_size[2],self.image_size[3]).unsqueeze(0)
         return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.tv = TotalVariation(reduction='mean')
   
    def forward(self,x,z):
        tv_loss=self.tv(x)
        x=x.squeeze().permute(0, 2, 1).reshape([ x.shape[1],-1])
        z=z.squeeze().permute(0, 2, 1).reshape([ z.shape[1],-1])
        residual1=torch.abs(z-x)
        loss2=torch.where(residual1< 1,0.5*(residual1**2),residual1-0.5)
        return loss2.sum()+0.05*tv_loss
    
    
class ASCR_Former(nn.Module):
    def __init__(self,patch_size,hid_local,abundance):
        super(ASCR_Former,self).__init__()
        input_size=abundance.shape
        self.abundance=abundance
        self.endmember=input_size[1]
        self.num_patches=(input_size[2]//patch_size[0])*(input_size[3]//patch_size[1])
        self.pixel_num=input_size[2]*input_size[3]
        self.U_net=ResCNN(self.endmember)
        self.Tokenization=Tokenization(patch_size)
        self.encode=T_Former(self.endmember, hidden_dim=hid_local, num_patches=self.num_patches,patch_pixel=patch_size[0]*patch_size[1])
        self.CAB=C_Former(endmember=self.endmember,num_patches=self.num_patches)
        self.SAB=S_Former(endmember=self.endmember,num_patches=self.num_patches)
        self.decode=nn.Sequential(Decoder(input_size,num_patches=self.num_patches),nn.Conv2d(self.endmember,self.endmember,3,padding=1))
        self.loss=Loss()

    def forward(self,x):
        x=self.U_net(x)
        x=self.Tokenization(x)
        x=self.encode(x).permute(1,0,2)
        x=self.SAB(x).permute(0,2,1)
        x=self.CAB(x)
        x=self.decode(x).softmax(dim=1)
        if self.training:
            return self.loss(x,self.abundance)
        else:
            return x

    

