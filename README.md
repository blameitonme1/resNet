# resNet
using pytorch to implement resNet
# resNet实现
基本借鉴d2l的课程代码，只不过我觉得还是要自己手打一遍才知道具体的设计是怎样的，所以就自己手动实现了一次，基本上和VGG块的设计理念很相似，
设计了一个由残差块组成的block，然后将这个block和全连接层，池化层组合起来就成了resNet。
具体设计如下  
![resnet18](https://github.com/blameitonme1/resNet/assets/113235913/450e8a09-8c59-4ea7-921c-336eaa38dadd)
