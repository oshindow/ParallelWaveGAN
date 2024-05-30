import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        
    def forward(self, x):
        padding = self.kernel_size - 1
        y = nn.functional.pad(x, (padding, 0))  # 在序列开头填充
        y_ = nn.functional.pad(x, (padding // 2, padding // 2))  # normal conv
        print(y, y_)
        return self.conv(y), self.conv(y_)

causalconv1d = CausalConv1d(in_channels=2, out_channels=3, kernel_size=1)
linear = nn.Linear(2,3)
tensor = torch.ones(1,2,10)
causalconvresult, normalconvresult = causalconv1d(tensor)
linearresult = linear(tensor.transpose(1,2))
print(tensor)
print(causalconv1d.conv._parameters)
print(causalconvresult)
print(normalconvresult)
print('########### linear ##############')
print(linear._parameters)
print(linearresult)
