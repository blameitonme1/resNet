import torch
from d2l import torch as d2l
from torch.nn import functional as F
from torch import nn
#  只有当较复杂的函数类包含较小的函数类时，我们才能确保提高它们的性能, 这是残差网络的思想
#  增加模型复杂度的同时，保证原来的函数被嵌套再现有的模型里面，这样就可以保证不会出现模型更加复杂但是
#  效果反而减弱的问题
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernerl_size=3, padding=1
        )
        if use_1x1conv:
            # 使用1x1的卷积层调整原输入的通道数好和卷积层的结果相加，适用于要修改输入的通道数时
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # batch normalisation layer,感觉中文很多还是使用英文好了,名字很直白
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, X):
        # 很直观的函数，之后resnet就是把这个残差块和别的全连接层之类的组合在一起即可
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
# 最初的block，和googlenet一样
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 每个除了第一个模块的模块1第一个残差块都会高宽减半，通道翻倍
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
# 还是默认在fashionMNIST上面训练，最后加上全局汇聚层和全连接层即可
net = nn.Sequential(b1, b2, b3, b4,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
# 训练一下看看效果,实际训练我还是在jupyter上面训练的, 方便一点
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())