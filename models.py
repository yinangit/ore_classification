import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/SunnyHaze/LeNet5-MNIST-Pytorch/blob/main/LeNet-5_GPU.py
class LeNet_5(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim,6,kernel_size=5,padding=2),# 原题为三通道，此处转为单通道实现 # C1
            nn.ReLU(),
            nn.MaxPool2d(2,2), # S2
            nn.Conv2d(6,16,5), # C3  原始论文中C3与S2并不是全连接而是部分连接，这样能减少部分计算量。而现代CNN模型中，比如AlexNet，ResNet等，都采取全连接的方式了。我们的实现在这里做了一些简化。
            nn.ReLU(),
            nn.MaxPool2d(2,2) # S4
        )
        # 然后需要经过变形后，继续进行全连接
        self.layer2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), # C5
            nn.ReLU(),
            nn.Linear(120, 84),         # F6
            nn.ReLU(),
            nn.Linear(84, n_class), # Output 文章中使用高斯连接，现在方便起见仍然使用全连接
        )
    def forward(self,x):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = x.view(-1,16 * 5 * 5) # 重新构建向量形状，准备全连接
        x = self.layer2(x) # 执行全连接部分
        return x

class dual_tower(nn.Module):
    def __init__(self, in_dim, n_class, pretrained=None) -> None:
        super(dual_tower, self).__init__()
        self.model_low = LeNet_5(in_dim, n_class)
        self.model_high = LeNet_5(in_dim, n_class)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            del params['layer2.4.weight']
            del params['layer2.4.bias']
            self.model_low.load_state_dict(params,strict=False)
            self.model_high.load_state_dict(params,strict=False)
    
    def forward(self, input_low, input_high):
        x_low = self.model_low(input_low)
        x_high = self.model_high(input_high)
        # fusion : sum of logit
        return x_low + x_high
    
class dual_tower_low(nn.Module):
    def __init__(self, in_dim, n_class, pretrained=None) -> None:
        super(dual_tower_low, self).__init__()
        self.model_low = LeNet_5(in_dim, n_class)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            del params['layer2.4.weight']
            del params['layer2.4.bias']
            self.model_low.load_state_dict(params,strict=False)
    
    def forward(self, input_low, input_high):
        x_low = self.model_low(input_low)
        return x_low
    
class dual_tower_high(nn.Module):
    def __init__(self, in_dim, n_class, pretrained=None) -> None:
        super(dual_tower_high, self).__init__()
        self.model_high = LeNet_5(in_dim, n_class)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            del params['layer2.4.weight']
            del params['layer2.4.bias']
            self.model_high.load_state_dict(params,strict=False)
    
    def forward(self, input_low, input_high):
        x_high = self.model_high(input_high)
        return x_high
    
class dual_tower_vis(nn.Module):
    def __init__(self, in_dim, n_class, pretrained=None) -> None:
        super(dual_tower_vis, self).__init__()
        self.model_low = LeNet_5(in_dim, n_class)
        self.model_high = LeNet_5(in_dim, n_class)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            del params['layer2.4.weight']
            del params['layer2.4.bias']
            self.model_low.load_state_dict(params,strict=False)
            self.model_high.load_state_dict(params,strict=False)
    
    def forward(self, inputs):
        input_low, input_high = inputs[:,0,:,:], inputs[:,1,:,:]
        input_low = input_low.unsqueeze(1)
        input_high = input_high.unsqueeze(1)
        x_low = self.model_low(input_low)
        x_high = self.model_high(input_high)
        # fusion : sum of logit
        return x_low + x_high


def create_model(args):
    assert args.modelName in ['dual_tower', 'dual_tower_high', 'dual_tower_low', 'dual_tower_vis']
    if args.modelName == 'dual_tower':
        return dual_tower(args.in_dim, args.nclass, args.pretrained)
    elif args.modelName == 'dual_tower_vis':
        return dual_tower_vis(args.in_dim, args.nclass, args.pretrained)
    elif args.modelName == 'dual_tower_high':
        return dual_tower_high(args.in_dim, args.nclass, args.pretrained)
    elif args.modelName == 'dual_tower_low':
        return dual_tower_low(args.in_dim, args.nclass, args.pretrained)
    else:
        raise NotImplementedError