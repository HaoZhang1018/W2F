import torch
import torch.nn as nn
import torch.nn.functional as F

class PointConv(nn.Module):
    """
    Point convolution block: input: x with size(B C H W); output size (B C1 H W)
    """
    def __init__(self, in_dim=64, out_dim=64, kernel_size=3, dilation=1, norm_layer=nn.BatchNorm2d):
        super(PointConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation       
        conv_padding = (self.kernel_size // 2) * self.dilation        
                
        self.pconv = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, padding=conv_padding, dilation=self.dilation),
                norm_layer(self.out_dim),
                nn.ReLU(inplace=True),
                )        

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.pconv(x) 
        return x


class CNNHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=16,
                 align_corners=False):        
        super(CNNHead, self).__init__()
        self.align_corners = align_corners        
        self.in_channels = in_channels
        F1_in_channels, F2_in_channels, F3_in_channels, F4_in_channels = self.in_channels
        embedding_dim = embed_dim
                
        self.PointConv1 = PointConv(in_dim=F1_in_channels, out_dim=embedding_dim, kernel_size=7, norm_layer=norm_layer)
        self.PointConv2 = PointConv(in_dim=F2_in_channels, out_dim=embedding_dim, kernel_size=5, norm_layer=norm_layer)
        self.PointConv3 = PointConv(in_dim=F3_in_channels, out_dim=embedding_dim, kernel_size=3, norm_layer=norm_layer)
        self.PointConv4 = PointConv(in_dim=F4_in_channels, out_dim=embedding_dim, kernel_size=1, norm_layer=norm_layer)
                
        self.CNN_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4+6, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=embedding_dim, out_channels=3, kernel_size=1),
                            nn.Sigmoid()
                            # nn.Tanh()
                            )
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        
        F0, F1, F2, F3, F4 = inputs

        F1_c = self.PointConv1(F1)
        F1_c = F.interpolate(F1_c, size=F0.size()[2:],mode='bilinear',align_corners=self.align_corners)
                      
        F2_c = self.PointConv2(F2)
        F2_c = F.interpolate(F2_c, size=F0.size()[2:], mode='bilinear',align_corners=self.align_corners)
               
        F3_c = self.PointConv3(F3)
        F3_c = F.interpolate(F3_c, size=F0.size()[2:],  mode='bilinear',align_corners=self.align_corners)
                
        F4_c = self.PointConv4(F4)
        F4_c = F.interpolate(F4_c, size=F0.size()[2:],  mode='bilinear',align_corners=self.align_corners)

        F_cat = torch.cat([F4_c, F3_c, F2_c, F1_c, F0], dim=1)
   
        Fuse = self.CNN_fuse(F_cat)
        # Fuse = Fuse / 2 + 0.5
        return Fuse
