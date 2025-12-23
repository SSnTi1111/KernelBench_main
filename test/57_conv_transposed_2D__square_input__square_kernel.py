import torch
import torch.nn as nn
# import torch.cuda.nvtx as nvtx

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# # Test code
# batch_size = 8
# in_channels = 64  # double channels for heavier compute
# out_channels = 64
# kernel_size = 3
# # larger square input
# height = 1024
# width = 1024
batch_size = 4          # 原来是 8，减半
in_channels = 64
out_channels = 64
kernel_size = 3
# larger square input
height = 512            # 原来是 1024 -> 显存占用降低到原来的 1/4
width = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = Model(in_channels, out_channels, kernel_size).cuda()
    x = torch.rand(batch_size, in_channels, height, width, device="cuda")

    # warmup（非常重要，避免测到初始化 kernel）
    for _ in range(5):
        y = model(x)
    torch.cuda.synchronize()

    print("开始 Profiling...")
    
    # 1. 通知 NCU 开始采集
    torch.cuda.cudart().cudaProfilerStart()

    # 2. 执行模型
    y = model(x)
    torch.cuda.synchronize() # 加上同步是个好习惯，保证录制完整

    # 3. 通知 NCU 停止采集
    torch.cuda.cudart().cudaProfilerStop()