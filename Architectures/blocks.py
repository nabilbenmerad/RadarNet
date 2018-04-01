import torch
import torch.nn as nn


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def soft_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def soft_conv_transpose(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=1, padding=1, bias=False)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)