import torch
import torch.nn as nn
from blocks import *

dataset_4x_1y_path = 'Data/64x64_non-norm/'
models_path = 'Models/'

class SoftConvDeconvEstimator(nn.Module):
    """
        The output channel size is the same as the input.
        This is done by removing the last two convolutional
        layers of FlowNetS, and adding two deconvolutional 
        layers at the end.
    """

    def __init__(self, input_channels=4, batch_norm=True):
        super(SoftConvDeconvEstimator, self).__init__()

        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.conv1 = conv(self.batch_norm, input_channels,
                          64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256, kernel_size=3)
        self.conv4 = conv(self.batch_norm, 256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512, kernel_size=3)
        self.conv5 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv5_1 = conv(self.batch_norm, 1024, 1024)

        self.deconv4 = soft_deconv(1024, 256)
        self.deconv3 = soft_deconv(512, 128)
        self.deconv2 = soft_deconv(386, 64)
        self.deconv1 = soft_deconv(194, 32)
        self.deconv0 = soft_deconv(98, 16)

        self.predict_flow5 = predict_flow(1024)
        self.predict_flow4 = predict_flow(256 + 512 + 2)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(98)
        self.predict_flow0 = predict_flow(input_channels + 18)

        self.upsampled_flow5_to_4 = soft_conv_transpose(2, 2)
        self.upsampled_flow4_to_3 = soft_conv_transpose(2, 2)
        self.upsampled_flow3_to_2 = soft_conv_transpose(2, 2)
        self.upsampled_flow2_to_1 = soft_conv_transpose(2, 2)

        self.conv_f = conv(self.batch_norm, 2, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # this modified initialization seems to work better, but it's
                # very hacky
                m.weight.data.normal_(0, 0.02 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # -> [908, 4, 64, 64] 
        out_conv1 = self.conv1(x) # -> [80, 64, 32, 32]
        out_conv2 = self.conv2(out_conv1) # -> [80, 128, 16, 16]
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) # -> [80, 256, 8, 8]
        out_conv4 = self.conv4_1(self.conv4(out_conv3)) # -> [80, 512, 4, 4]
        
        out_conv5 = self.conv5_1(self.conv5(out_conv4)) # -> [80, 1024, 2, 2]
        out_deconv4 = self.deconv4(out_conv5) # -> [80, 256, 4, 4]
        flow5 = self.predict_flow5(out_conv5) # -> [80, 2, 2, 2]
        flow5_up = self.upsampled_flow5_to_4(flow5) # -> [80, 2, 4, 4]
        
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1) # -> [80, 770, 4, 4]
        out_deconv3 = self.deconv3(out_conv4) # -> [80, 128, 8, 8]
        flow4 = self.predict_flow4(concat4) # -> [80, 2, 4, 4]
        flow4_up = self.upsampled_flow4_to_3(flow4) # -> [80, 2, 8, 8]
     
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1) # -> [80, 386, 8, 8]
        out_deconv2 = self.deconv2(concat3) # -> [80, 64, 16, 16]
        flow3 = self.predict_flow3(concat3) # -> [80, 2, 8, 8]
        flow3_up = self.upsampled_flow3_to_2(flow3) # -> [80, 2, 16, 16]

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1) # -> [80, 194, 16, 16]
        out_deconv1 = self.deconv1(concat2) # -> [80, 32, 32, 32]
        flow2 = self.predict_flow2(concat2) # -> [80, 2, 16, 16]
        flow2_up = self.upsampled_flow2_to_1(flow2) # -> [80, 2, 32, 32]

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1) # -> [80, 98, 32, 32]
        out_deconv0 = self.deconv0(concat1) # -> [80, 16, 64, 64]
        flow1 = self.predict_flow1(concat1) # -> [80, 2, 32, 32]
        flow1_up = self.upsampled_flow2_to_1(flow1) # -> [80, 2, 64, 64]

        concat0 = torch.cat((x, out_deconv0, flow1_up), 1) # -> [80, 22, 64, 64]
        flow0 = self.predict_flow0(concat0) # -> [80, 2, 64, 64]

        return flow0
        #return self.conv_f(flow0)
