! pip install einops

import numpy as np # linear algebra
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm

img_size = 64
patch_h = 8
patch_w = 8
n_patch = img_size//patch_h*img_size//patch_w
channels = 3
patch_dim = patch_h * patch_w *channels
dim = 1024
batch_size = 3

n_heads = 8
n_layers = 8
dict_q = dict_k = dict_v = 1024
dropout = 0.1

d_ff = 2048

n_classes = 20

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,dim),
            nn.Dropout(dropout)
        )
        
    def forward(self,x):#x.size:[batch, n, dim]
        return self.net(x)
    
class MultHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(dim, dict_q*n_heads, bias=False)
        self.W_k = nn.Linear(dim, dict_k*n_heads, bias=False)
        self.W_v = nn.Linear(dim, dict_v*n_heads, bias=False)
        self.project = nn.Linear(n_heads*dict_v, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):#x.size:[batch, n, dim]
        batch = x.shape[0]
        q = self.W_q(x).view(batch, -1, n_heads, dict_q).transpose(1,2)
        k = self.W_k(x).view(batch, -1, n_heads, dict_k).transpose(1,2)
        v = self.W_v(x).view(batch, -1, n_heads, dict_v).transpose(1,2)
        core = torch.matmul(q, k.transpose(2,3))*dict_k ** -0.5
        attn = nn.Softmax(dim=-1)(core)
        #x.shape:[b, heads, n, n]
        x = torch.matmul(attn, v)
        x = x.transpose(1,2).contiguous().view(batch, -1, n_heads*dict_v)
        out = self.dropout((self.project(x)))
        return out

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.atttion = MultHeadAttention()
        self.ff = FeedForward()
        
    def forward(self,x):#x.size:[batch, n, dim]
        residual = x
        x = self.norm(x)
        x = self.atttion(x) + residual
        
        residual = x
        x = self.norm(x)
        x = self.ff(x) + residual
        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for layer in range(n_layers)])
                
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn([1,1,dim]))
        self.pos_emb = nn.Parameter(torch.randn(1, n_patch+1,dim))
        self.transformer = Transformer()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )
        
    def forward(self,img):
        #img.size:[b, c, img_h, img_w]
        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_h, p2 = patch_w)
        x = self.emb(x)
        b, n, _ = x.shape
        #x.shape:[b, n, dim]
        
        cls_tokens = repeat(self.cls_token, '() n dim -> b n dim', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = repeat(self.pos_emb, "() n d -> b n d", b = b)
        x = x + pos
        
        x = self.transformer(x)
        out = x.mean(dim=1) if False else x[:,0]
        
        return self.mlp_head(out)
        
    
if __name__ =="__main__":
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
        
    trainset = torchvision.datasets.CIFAR10(root='/kaggle/working/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    lr = 1e-4
    epochs = 10
    model = Vit().to(device)
    critizer = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        losses = 0
        for i, data in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            x, y = data
            x= x.to(device)
            y = y.to(device)
            output = model(x)
            loss = critizer(output, y)
            loss.backward()
            losses += loss.item()
            optimizer.step()
            if i % 100 == 0:
                print(f'loss:{loss:.2f}')
        torch.save(model.state_dict(), f'/kaggle/working/vit{epoch+1}.pt')
        print(f'epoch:{epoch}, loss:{loss/len(trainloader):.2f}')
