import numpy as np
import mindspore
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_1 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_sigmoid_1 = self.sigmoid_1(opt_conv2d_0)
        opt_mul_2 = P.Mul()(opt_conv2d_0, opt_sigmoid_1)
        return opt_mul_2


class Module12(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode):
        super(Module12, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        return module1_1_opt


class Module11(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_2_conv2d_0_in_channels, module1_2_conv2d_0_out_channels, module1_2_conv2d_0_kernel_size,
                 module1_2_conv2d_0_stride, module1_2_conv2d_0_padding, module1_2_conv2d_0_pad_mode):
        super(Module11, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)
        self.module1_2 = Module1(conv2d_0_in_channels=module1_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_2_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_2_conv2d_0_stride,
                                 conv2d_0_padding=module1_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_2_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        module1_2_opt = self.module1_2(module1_1_opt)
        opt_add_0 = P.Add()(module1_0_opt, module1_2_opt)
        return opt_add_0


class Module15(nn.Cell):
    def __init__(self, module1_0_conv2d_0_in_channels, module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size,
                 module1_0_conv2d_0_stride, module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode,
                 module1_1_conv2d_0_in_channels, module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size,
                 module1_1_conv2d_0_stride, module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode,
                 module1_2_conv2d_0_in_channels, module1_2_conv2d_0_out_channels, module1_2_conv2d_0_kernel_size,
                 module1_2_conv2d_0_stride, module1_2_conv2d_0_padding, module1_2_conv2d_0_pad_mode):
        super(Module15, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)
        self.module1_2 = Module1(conv2d_0_in_channels=module1_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_2_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_2_conv2d_0_stride,
                                 conv2d_0_padding=module1_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_2_conv2d_0_pad_mode)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        module1_2_opt = self.module1_2(module1_1_opt)
        return module1_2_opt


class Module8(nn.Cell):
    def __init__(self, stridedslice_0_begin, stridedslice_0_end):
        super(Module8, self).__init__()
        self.stridedslice_0 = P.StridedSlice()
        self.stridedslice_0_begin = stridedslice_0_begin
        self.stridedslice_0_end = stridedslice_0_end
        self.stridedslice_0_strides = (1, 1, 1, 1, 1)
        self.mul_1_w = 2.0

    def construct(self, x):
        opt_stridedslice_0 = self.stridedslice_0(x, self.stridedslice_0_begin, self.stridedslice_0_end,
                                                 self.stridedslice_0_strides)
        opt_mul_1 = opt_stridedslice_0 * self.mul_1_w
        return opt_mul_1


class Model(nn.Cell):
    def __init__(self, img_size=1024, bs=1):  # bs - batch_size
        super(Model, self).__init__()

        self.stridedslice_0 = P.StridedSlice()
        self.stridedslice_0_begin = (0, 0, 0, 0)
        self.stridedslice_0_end = (bs, 3, img_size, img_size)
        self.stridedslice_0_strides = (1, 1, 2, 1)
        self.stridedslice_4 = P.StridedSlice()
        self.stridedslice_4_begin = (0, 0, 0, 0)
        self.stridedslice_4_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_4_strides = (1, 1, 1, 2)
        self.stridedslice_1 = P.StridedSlice()
        self.stridedslice_1_begin = (0, 0, 1, 0)
        self.stridedslice_1_end = (bs, 3, img_size, img_size)
        self.stridedslice_1_strides = (1, 1, 2, 1)
        self.stridedslice_5 = P.StridedSlice()
        self.stridedslice_5_begin = (0, 0, 0, 0)
        self.stridedslice_5_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_5_strides = (1, 1, 1, 2)
        self.stridedslice_2 = P.StridedSlice()
        self.stridedslice_2_begin = (0, 0, 0, 0)
        self.stridedslice_2_end = (bs, 3, img_size, img_size)
        self.stridedslice_2_strides = (1, 1, 2, 1)
        self.stridedslice_6 = P.StridedSlice()
        self.stridedslice_6_begin = (0, 0, 0, 1)
        self.stridedslice_6_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_6_strides = (1, 1, 1, 2)
        self.stridedslice_3 = P.StridedSlice()
        self.stridedslice_3_begin = (0, 0, 1, 0)
        self.stridedslice_3_end = (bs, 3, img_size, img_size)
        self.stridedslice_3_strides = (1, 1, 2, 1)
        self.stridedslice_7 = P.StridedSlice()
        self.stridedslice_7_begin = (0, 0, 0, 1)
        self.stridedslice_7_end = (bs, 3, img_size//2, img_size)
        self.stridedslice_7_strides = (1, 1, 1, 2)
        self.concat_8 = P.Concat(axis=1)
        self.module12_0 = Module12(module1_0_conv2d_0_in_channels=12,
                                   module1_0_conv2d_0_out_channels=32,
                                   module1_0_conv2d_0_kernel_size=(3, 3),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_0_conv2d_0_pad_mode="pad",
                                   module1_1_conv2d_0_in_channels=32,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(2, 2),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module11_0 = Module11(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=32,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=32,
                                   module1_1_conv2d_0_out_channels=32,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=32,
                                   module1_2_conv2d_0_out_channels=32,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_0 = Module1(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_28 = P.Concat(axis=1)
        self.module12_1 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(2, 2),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module11_1 = Module11(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=64,
                                   module1_2_conv2d_0_out_channels=64,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module12_2 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module12_3 = Module12(module1_0_conv2d_0_in_channels=64,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_1 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_62 = P.Concat(axis=1)
        self.module1_2 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_3 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module11_2 = Module11(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module12_4 = Module12(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module12_5 = Module12(module1_0_conv2d_0_in_channels=128,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_4 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.concat_96 = P.Concat(axis=1)
        self.module1_5 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_6 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_7 = Module1(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.pad_maxpool2d_106 = nn.Pad(paddings=((0, 0), (0, 0), (2, 2), (2, 2)))
        self.maxpool2d_106 = nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.pad_maxpool2d_107 = nn.Pad(paddings=((0, 0), (0, 0), (4, 4), (4, 4)))
        self.maxpool2d_107 = nn.MaxPool2d(kernel_size=(9, 9), stride=(1, 1))
        self.pad_maxpool2d_108 = nn.Pad(paddings=((0, 0), (0, 0), (6, 6), (6, 6)))
        self.maxpool2d_108 = nn.MaxPool2d(kernel_size=(13, 13), stride=(1, 1))
        self.concat_109 = P.Concat(axis=1)
        self.module1_8 = Module1(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module1_9 = Module1(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module12_6 = Module12(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(3, 3),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_1_conv2d_0_pad_mode="pad")
        self.module1_10 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_125 = P.Concat(axis=1)
        self.module12_7 = Module12(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=512,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=512,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid")
        self.resizenearestneighbor_132 = P.ResizeNearestNeighbor(size=(img_size//16, img_size//16))
        self.concat_133 = P.Concat(axis=1)
        self.module15_0 = Module15(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_11 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_146 = P.Concat(axis=1)
        self.module12_8 = Module12(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid")
        self.resizenearestneighbor_153 = P.ResizeNearestNeighbor(size=(img_size//8, img_size//8))
        self.concat_154 = P.Concat(axis=1)
        self.module15_1 = Module15(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=64,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=64,
                                   module1_1_conv2d_0_out_channels=64,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=64,
                                   module1_2_conv2d_0_out_channels=64,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_12 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_167 = P.Concat(axis=1)
        self.module1_13 = Module1(conv2d_0_in_channels=128,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.module1_14 = Module1(conv2d_0_in_channels=128,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(2, 2),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.concat_177 = P.Concat(axis=1)
        self.module15_2 = Module15(module1_0_conv2d_0_in_channels=256,
                                   module1_0_conv2d_0_out_channels=128,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=128,
                                   module1_1_conv2d_0_out_channels=128,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=128,
                                   module1_2_conv2d_0_out_channels=128,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_15 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_203 = P.Concat(axis=1)
        self.module1_16 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.module1_17 = Module1(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(2, 2),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.concat_213 = P.Concat(axis=1)
        self.module15_3 = Module15(module1_0_conv2d_0_in_channels=512,
                                   module1_0_conv2d_0_out_channels=256,
                                   module1_0_conv2d_0_kernel_size=(1, 1),
                                   module1_0_conv2d_0_stride=(1, 1),
                                   module1_0_conv2d_0_padding=0,
                                   module1_0_conv2d_0_pad_mode="valid",
                                   module1_1_conv2d_0_in_channels=256,
                                   module1_1_conv2d_0_out_channels=256,
                                   module1_1_conv2d_0_kernel_size=(1, 1),
                                   module1_1_conv2d_0_stride=(1, 1),
                                   module1_1_conv2d_0_padding=0,
                                   module1_1_conv2d_0_pad_mode="valid",
                                   module1_2_conv2d_0_in_channels=256,
                                   module1_2_conv2d_0_out_channels=256,
                                   module1_2_conv2d_0_kernel_size=(3, 3),
                                   module1_2_conv2d_0_stride=(1, 1),
                                   module1_2_conv2d_0_padding=(1, 1, 1, 1),
                                   module1_2_conv2d_0_pad_mode="pad")
        self.module1_18 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.concat_239 = P.Concat(axis=1)
        self.module1_19 = Module1(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=512,
                                  conv2d_0_kernel_size=(1, 1),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=0,
                                  conv2d_0_pad_mode="valid")
        self.conv2d_172 = nn.Conv2d(in_channels=128,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_174 = P.Reshape()
        self.reshape_174_shape = tuple([bs, 3, 11, img_size//8, img_size//8])
        self.transpose_176 = P.Transpose()
        self.sigmoid_178 = nn.Sigmoid()
        self.module8_0 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//8, img_size//8, 2))
        self.sub_190_bias = 0.5
        self.add_193_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//8, img_size//8, 2)).astype(np.float32)),
                                      name=None)
        self.mul_196_w = 8.0
        self.module8_1 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//8, img_size//8, 4))
        self.pow_191 = P.Pow()
        self.pow_191_input_weight = 2.0
        self.mul_194_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_183 = P.StridedSlice()
        self.stridedslice_183_begin = (0, 0, 0, 0, 4)
        self.stridedslice_183_end = (bs, 3, img_size//8, img_size//8, 11)
        self.stridedslice_183_strides = (1, 1, 1, 1, 1)
        self.concat_198 = P.Concat(axis=-1)
        self.reshape_200 = P.Reshape()
        self.reshape_200_shape = tuple([bs, 3*(img_size//8)*(img_size//8), 11])
        self.conv2d_208 = nn.Conv2d(in_channels=256,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_210 = P.Reshape()
        self.reshape_210_shape = tuple([bs, 3, 11, img_size//16, img_size//16])
        self.transpose_212 = P.Transpose()
        self.sigmoid_214 = nn.Sigmoid()
        self.module8_2 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//16, img_size//16, 2))
        self.sub_226_bias = 0.5
        self.add_229_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//16, img_size//16, 2)).astype(np.float32)), name=None)
        self.mul_232_w = 16.0
        self.module8_3 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//16, img_size//16, 4))
        self.pow_227 = P.Pow()
        self.pow_227_input_weight = 2.0
        self.mul_230_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_219 = P.StridedSlice()
        self.stridedslice_219_begin = (0, 0, 0, 0, 4)
        self.stridedslice_219_end = (bs, 3, img_size//16, img_size//16, 11)
        self.stridedslice_219_strides = (1, 1, 1, 1, 1)
        self.concat_234 = P.Concat(axis=-1)
        self.reshape_236 = P.Reshape()
        self.reshape_236_shape = tuple([bs, 3*(img_size//16)*(img_size//16), 11])
        self.conv2d_243 = nn.Conv2d(in_channels=512,
                                    out_channels=33,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.reshape_244 = P.Reshape()
        self.reshape_244_shape = tuple([bs, 3, 11, img_size//32, img_size//32])
        self.transpose_245 = P.Transpose()
        self.sigmoid_246 = nn.Sigmoid()
        self.module8_4 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 0), stridedslice_0_end=(bs, 3, img_size//32, img_size//32, 2))
        self.sub_252_bias = 0.5
        self.add_254_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, img_size//32, img_size//32, 2)).astype(np.float32)), name=None)
        self.mul_256_w = 32.0
        self.module8_5 = Module8(stridedslice_0_begin=(0, 0, 0, 0, 2), stridedslice_0_end=(bs, 3, img_size//32, img_size//32, 4))
        self.pow_253 = P.Pow()
        self.pow_253_input_weight = 2.0
        self.mul_255_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 3, 1, 1, 2)).astype(np.float32)), name=None)
        self.stridedslice_249 = P.StridedSlice()
        self.stridedslice_249_begin = (0, 0, 0, 0, 4)
        self.stridedslice_249_end = (bs, 3, img_size//32, img_size//32, 11)
        self.stridedslice_249_strides = (1, 1, 1, 1, 1)
        self.concat_257 = P.Concat(axis=-1)
        self.reshape_258 = P.Reshape()
        self.reshape_258_shape = tuple([bs, 3*(img_size//32)*(img_size//32), 11])
        self.concat_259 = P.Concat(axis=1)

    def construct(self, images):
        opt_stridedslice_0 = self.stridedslice_0(images, self.stridedslice_0_begin, self.stridedslice_0_end,
                                                 self.stridedslice_0_strides)
        opt_stridedslice_4 = self.stridedslice_4(opt_stridedslice_0, self.stridedslice_4_begin, self.stridedslice_4_end,
                                                 self.stridedslice_4_strides)
        opt_stridedslice_1 = self.stridedslice_1(images, self.stridedslice_1_begin, self.stridedslice_1_end,
                                                 self.stridedslice_1_strides)
        opt_stridedslice_5 = self.stridedslice_5(opt_stridedslice_1, self.stridedslice_5_begin, self.stridedslice_5_end,
                                                 self.stridedslice_5_strides)
        opt_stridedslice_2 = self.stridedslice_2(images, self.stridedslice_2_begin, self.stridedslice_2_end,
                                                 self.stridedslice_2_strides)
        opt_stridedslice_6 = self.stridedslice_6(opt_stridedslice_2, self.stridedslice_6_begin, self.stridedslice_6_end,
                                                 self.stridedslice_6_strides)
        opt_stridedslice_3 = self.stridedslice_3(images, self.stridedslice_3_begin, self.stridedslice_3_end,
                                                 self.stridedslice_3_strides)
        opt_stridedslice_7 = self.stridedslice_7(opt_stridedslice_3, self.stridedslice_7_begin, self.stridedslice_7_end,
                                                 self.stridedslice_7_strides)
        opt_concat_8 = self.concat_8((opt_stridedslice_4, opt_stridedslice_5, opt_stridedslice_6, opt_stridedslice_7, ))
        module12_0_opt = self.module12_0(opt_concat_8)
        module11_0_opt = self.module11_0(module12_0_opt)
        module1_0_opt = self.module1_0(module12_0_opt)
        opt_concat_28 = self.concat_28((module11_0_opt, module1_0_opt, ))
        module12_1_opt = self.module12_1(opt_concat_28)
        module11_1_opt = self.module11_1(module12_1_opt)
        module12_2_opt = self.module12_2(module11_1_opt)
        opt_add_54 = P.Add()(module11_1_opt, module12_2_opt)
        module12_3_opt = self.module12_3(opt_add_54)
        opt_add_61 = P.Add()(opt_add_54, module12_3_opt)
        module1_1_opt = self.module1_1(module12_1_opt)
        opt_concat_62 = self.concat_62((opt_add_61, module1_1_opt, ))
        module1_2_opt = self.module1_2(opt_concat_62)
        module1_3_opt = self.module1_3(module1_2_opt)
        module11_2_opt = self.module11_2(module1_3_opt)
        module12_4_opt = self.module12_4(module11_2_opt)
        opt_add_88 = P.Add()(module11_2_opt, module12_4_opt)
        module12_5_opt = self.module12_5(opt_add_88)
        opt_add_95 = P.Add()(opt_add_88, module12_5_opt)
        module1_4_opt = self.module1_4(module1_3_opt)
        opt_concat_96 = self.concat_96((opt_add_95, module1_4_opt, ))
        module1_5_opt = self.module1_5(opt_concat_96)
        module1_6_opt = self.module1_6(module1_5_opt)
        module1_7_opt = self.module1_7(module1_6_opt)
        opt_maxpool2d_106 = self.pad_maxpool2d_106(module1_7_opt)
        opt_maxpool2d_106 = self.maxpool2d_106(opt_maxpool2d_106)
        opt_maxpool2d_107 = self.pad_maxpool2d_107(module1_7_opt)
        opt_maxpool2d_107 = self.maxpool2d_107(opt_maxpool2d_107)
        opt_maxpool2d_108 = self.pad_maxpool2d_108(module1_7_opt)
        opt_maxpool2d_108 = self.maxpool2d_108(opt_maxpool2d_108)
        opt_concat_109 = self.concat_109((module1_7_opt, opt_maxpool2d_106, opt_maxpool2d_107, opt_maxpool2d_108, ))
        module1_8_opt = self.module1_8(opt_concat_109)
        module1_9_opt = self.module1_9(module1_8_opt)
        module12_6_opt = self.module12_6(module1_9_opt)
        module1_10_opt = self.module1_10(module1_8_opt)
        opt_concat_125 = self.concat_125((module12_6_opt, module1_10_opt, ))
        module12_7_opt = self.module12_7(opt_concat_125)
        opt_resizenearestneighbor_132 = self.resizenearestneighbor_132(module12_7_opt)
        opt_concat_133 = self.concat_133((opt_resizenearestneighbor_132, module1_5_opt, ))
        module15_0_opt = self.module15_0(opt_concat_133)
        module1_11_opt = self.module1_11(opt_concat_133)
        opt_concat_146 = self.concat_146((module15_0_opt, module1_11_opt, ))
        module12_8_opt = self.module12_8(opt_concat_146)
        opt_resizenearestneighbor_153 = self.resizenearestneighbor_153(module12_8_opt)
        opt_concat_154 = self.concat_154((opt_resizenearestneighbor_153, module1_2_opt, ))
        module15_1_opt = self.module15_1(opt_concat_154)
        module1_12_opt = self.module1_12(opt_concat_154)
        opt_concat_167 = self.concat_167((module15_1_opt, module1_12_opt, ))
        module1_13_opt = self.module1_13(opt_concat_167)
        module1_14_opt = self.module1_14(module1_13_opt)
        opt_concat_177 = self.concat_177((module1_14_opt, module12_8_opt, ))
        module15_2_opt = self.module15_2(opt_concat_177)
        module1_15_opt = self.module1_15(opt_concat_177)
        opt_concat_203 = self.concat_203((module15_2_opt, module1_15_opt, ))
        module1_16_opt = self.module1_16(opt_concat_203)
        module1_17_opt = self.module1_17(module1_16_opt)
        opt_concat_213 = self.concat_213((module1_17_opt, module12_7_opt, ))
        module15_3_opt = self.module15_3(opt_concat_213)
        module1_18_opt = self.module1_18(opt_concat_213)
        opt_concat_239 = self.concat_239((module15_3_opt, module1_18_opt, ))
        module1_19_opt = self.module1_19(opt_concat_239)
        opt_conv2d_172 = self.conv2d_172(module1_13_opt)
        opt_reshape_174 = self.reshape_174(opt_conv2d_172, self.reshape_174_shape)
        opt_transpose_176 = self.transpose_176(opt_reshape_174, (0, 1, 3, 4, 2))
        opt_sigmoid_178 = self.sigmoid_178(opt_transpose_176)
        module8_0_opt = self.module8_0(opt_sigmoid_178)
        opt_sub_190 = module8_0_opt - self.sub_190_bias
        opt_add_193 = opt_sub_190 + self.add_193_bias
        opt_mul_196 = opt_add_193 * self.mul_196_w
        module8_1_opt = self.module8_1(opt_sigmoid_178)
        opt_pow_191 = self.pow_191(module8_1_opt, self.pow_191_input_weight)
        opt_mul_194 = opt_pow_191 * self.mul_194_w
        opt_stridedslice_183 = self.stridedslice_183(opt_sigmoid_178, self.stridedslice_183_begin,
                                                     self.stridedslice_183_end, self.stridedslice_183_strides)
        opt_concat_198 = self.concat_198((opt_mul_196, opt_mul_194, opt_stridedslice_183, ))
        opt_reshape_200 = self.reshape_200(opt_concat_198, self.reshape_200_shape)
        opt_conv2d_208 = self.conv2d_208(module1_16_opt)
        opt_reshape_210 = self.reshape_210(opt_conv2d_208, self.reshape_210_shape)
        opt_transpose_212 = self.transpose_212(opt_reshape_210, (0, 1, 3, 4, 2))
        opt_sigmoid_214 = self.sigmoid_214(opt_transpose_212)
        module8_2_opt = self.module8_2(opt_sigmoid_214)
        opt_sub_226 = module8_2_opt - self.sub_226_bias
        opt_add_229 = opt_sub_226 + self.add_229_bias
        opt_mul_232 = opt_add_229 * self.mul_232_w
        module8_3_opt = self.module8_3(opt_sigmoid_214)
        opt_pow_227 = self.pow_227(module8_3_opt, self.pow_227_input_weight)
        opt_mul_230 = opt_pow_227 * self.mul_230_w
        opt_stridedslice_219 = self.stridedslice_219(opt_sigmoid_214, self.stridedslice_219_begin,
                                                     self.stridedslice_219_end, self.stridedslice_219_strides)
        opt_concat_234 = self.concat_234((opt_mul_232, opt_mul_230, opt_stridedslice_219, ))
        opt_reshape_236 = self.reshape_236(opt_concat_234, self.reshape_236_shape)
        opt_conv2d_243 = self.conv2d_243(module1_19_opt)
        opt_reshape_244 = self.reshape_244(opt_conv2d_243, self.reshape_244_shape)
        opt_transpose_245 = self.transpose_245(opt_reshape_244, (0, 1, 3, 4, 2))
        opt_sigmoid_246 = self.sigmoid_246(opt_transpose_245)
        module8_4_opt = self.module8_4(opt_sigmoid_246)
        opt_sub_252 = module8_4_opt - self.sub_252_bias
        opt_add_254 = opt_sub_252 + self.add_254_bias
        opt_mul_256 = opt_add_254 * self.mul_256_w
        module8_5_opt = self.module8_5(opt_sigmoid_246)
        opt_pow_253 = self.pow_253(module8_5_opt, self.pow_253_input_weight)
        opt_mul_255 = opt_pow_253 * self.mul_255_w
        opt_stridedslice_249 = self.stridedslice_249(opt_sigmoid_246, self.stridedslice_249_begin,
                                                     self.stridedslice_249_end, self.stridedslice_249_strides)
        opt_concat_257 = self.concat_257((opt_mul_256, opt_mul_255, opt_stridedslice_249, ))
        opt_reshape_258 = self.reshape_258(opt_concat_257, self.reshape_258_shape)
        preds = self.concat_259((opt_reshape_200, opt_reshape_236, opt_reshape_258, ))
        return opt_transpose_176, opt_transpose_212, opt_transpose_245, preds
