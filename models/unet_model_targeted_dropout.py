# Nicola Dinsdale 2020
# Model for unlearning domain for segmentation task
########################################################################################################################
# Import dependencies
from collections import OrderedDict

import torch
import torch.nn as nn
########################################################################################################################

class UNet1(nn.Module):

    def __init__(self, drop_probs, in_channels=1, init_features=4):
        super(UNet1, self).__init__()

        self.drop_probs = drop_probs

        features = init_features
        self.encoder1 = UNet1._block(self.drop_probs, 0, 1, in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet1._block(self.drop_probs, 2, 3, features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet1._block(self.drop_probs, 4, 5, features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet1._block(self.drop_probs, 6, 7, features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet1._block(self.drop_probs, 8, 9, features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1._block(self.drop_probs, 10, 11, (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1._block(self.drop_probs, 12, 13, (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1._block(self.drop_probs, 14, 15, (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1._half_block(self.drop_probs, 16, features * 2, features, name="dec1")



    def forward(self, x):
        
        # Process encoder1 layer by layer instead of a single call.
        # enc1_output = x
        # # Assuming encoder1 is a nn.Sequential; otherwise, adjust accordingly.
        # for idx, layer in enumerate(self.encoder1):
        #     enc1_output = layer(enc1_output)
            #print(f"Encoder1 - Layer {idx}: Shape: {enc1_output.shape}")
            #non_zero = enc1_output[enc1_output != 0] if torch.sum(enc1_output != 0) > 0 else enc1_output
            #print(f"Encoder1 - Layer {idx} stats: Min: {torch.min(enc1_output).item()}, Max: {torch.max(enc1_output).item()}, Mean: {torch.mean(non_zero).item()}")
            #print(f"Encoder1 - Layer {idx}: Number of Zeros: {torch.sum(enc1_output == 0).item()}, Total elements: {enc1_output.numel()}")

            #if idx == 2:
                #print the running stats of the batchnorm layer
                #print("Encoder1 - Layer 2 - BatchNorm running stats: ", layer.running_mean, layer.running_var)
                #print("Encoder1 - Layer 2 - BatchNorm weight: ", layer.weight)
                #print("Encoder1 - Layer 2 - BatchNorm bias: ", layer.bias)
            
        #extras.append(enc1_output)
        
        # If other_extras is provided, use its first element instead of new enc1_output.
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        #extras.append(enc4)

        #print("Shape of enc4 output: ", enc4.shape)

        # each = enc4
        # print("Min: ", torch.min(each).item(), "Max: ", torch.max(each).item(), "Mean: ", torch.mean(each[each != 0]).item())
        #             #number of zeros in the intermediate output
        # print("Number of zeros: ", torch.sum(each == 0).item())
        # print("Total number of elements: ", each.numel())


        #enc4 = enc4 if not other_extras else other_extras[1]
        bottleneck = self.bottleneck(self.pool4(enc4))
        #print("Shape of bottleneck output: ", bottleneck.shape)
        
        # each = bottleneck
        # print("Min: ", torch.min(each).item(), "Max: ", torch.max(each).item(), "Mean: ", torch.mean(each[each != 0]).item())
        #             #number of zeros in the intermediate output
        # print("Number of zeros: ", torch.sum(each == 0).item())
        # print("Total number of elements: ", each.numel())

        # extras.append(bottleneck)

        # bottleneck = bottleneck if not other_extras else other_extras[2]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # extras.append(dec1)
        # dec1 = dec1 if not other_extras else other_extras[4]
        return dec1

    @staticmethod
    def _block(drop_probs, key1, key2, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop1', torch.nn.Dropout3d(p=drop_probs[key1])),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop2', torch.nn.Dropout3d(p=drop_probs[key2])),
                ]
            )
        )

    @staticmethod
    def _half_block(drop_probs, key1, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop1', torch.nn.Dropout3d(p=drop_probs[key1])),
                ]
            )
        )

class Segmenter(nn.Module):
    def __init__(self, init_features=4, drop_prob=0.1):
        super(Segmenter, self).__init__()
        features = init_features

        self.decoder1 = Segmenter._half_block(features, features, name="dec1", drop_prob=drop_prob)
        self.conv = nn.Conv3d(in_channels=features, out_channels=1, kernel_size=1)

    def forward(self, x):
        dec1 = self.decoder1(x)
        bottleneck = self.conv(dec1)
        return torch.sigmoid(bottleneck), bottleneck

    @staticmethod
    def _half_block(in_channels, features, name, drop_prob):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "drop1", nn.Dropout3d(p=drop_prob)),
                ]
            )
        )
    

class UNet5_old(nn.Module):
    def __init__(self, drop_probs, in_channels=1, init_features=8):
        super(UNet5_old, self).__init__()
        features = init_features
        
        # Define dropout probabilities for each layer
        self.drop_probs = drop_probs
        
        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1", drop_probs=self.drop_probs, key1=0, key2=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2", drop_probs=self.drop_probs, key1=2, key2=3)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3", drop_probs=self.drop_probs, key1=4, key2=5)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4", drop_probs=self.drop_probs, key1=6, key2=7)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = self._block(features * 8, features * 16, name="enc5", drop_probs=self.drop_probs, key1=8, key2=9)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 16, features * 32, name="bottleneck", drop_probs=self.drop_probs, key1=10, key2=11)
        
        # Decoder path
        self.upconv5 = nn.ConvTranspose3d(
            features * 32, features * 16, kernel_size=2, stride=2
        )
        self.decoder5 = self._block(features * 32, features * 16, name="dec5", drop_probs=self.drop_probs, key1=12, key2=13)
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block(features * 16, features * 8, name="dec4", drop_probs=self.drop_probs, key1=14, key2=15)
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block(features * 8, features * 4, name="dec3", drop_probs=self.drop_probs, key1=16, key2=17)
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block(features * 4, features * 2, name="dec2", drop_probs=self.drop_probs, key1=18, key2=19)
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._half_block(features * 2, features, name="dec1", drop_probs=self.drop_probs, key1=20)

    @staticmethod
    def _block(in_channels, features, name, drop_probs, key1, key2):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop1', torch.nn.Dropout3d(p=drop_probs[key1])),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop2', torch.nn.Dropout3d(p=drop_probs[key2])),
                ]
            )
        )

    @staticmethod
    def _half_block(in_channels, features, name, drop_probs, key1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + 'drop1', torch.nn.Dropout3d(p=drop_probs[key1])),
                ]
            )
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))
        
        # Decoder with skip connections
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return dec1


IN = 0
OUT = 1

class UNet5(nn.Module):
    def __init__(self, in_channels=1, init_features=16, drop_prob=0.1):
        super(UNet5, self).__init__()
        self.drop_prob = drop_prob
        self.drop_probs = {}
        self._init_dropout_probs(drop_prob)
        features = [init_features, 2*init_features, 4*init_features, 8*init_features, 16*init_features, 32*init_features]
        self.features = features
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.ignore_for_pruning = set(["decoder1conv2"])

        self.encoder1 = self._double_conv_block(in_channels, features[0], "encoder1")
        self.encoder2 = self._double_conv_block(features[0], features[1], "encoder2")
        self.encoder3 = self._double_conv_block(features[1], features[2], "encoder3")
        self.encoder4 = self._double_conv_block(features[2], features[3], "encoder4")
        self.encoder5 = self._double_conv_block(features[3], features[4], "encoder5")

        self.bottleneck = self._double_conv_block(features[4], features[5], "bottleneck")

        self.upconv5 = nn.ConvTranspose3d(features[5], features[4], kernel_size=2, stride=2)
        self.decoder5 = self._double_conv_block(features[4]*2, features[4], "decoder5")

        self.upconv4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = self._double_conv_block(features[3]*2, features[3], "decoder4")

        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = self._double_conv_block(features[2]*2, features[2], "decoder3")

        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._double_conv_block(features[1]*2, features[1], "decoder2")

        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._double_conv_block(features[0]*2, features[0], "decoder1")

        self.final_conv = nn.Conv3d(features[0], 1, kernel_size=1)
        self.final_conv.name = "final_conv"

        self.filter_dependencies = {} #creates a mapping from layer name to the layers that are affected when a filter is removed from the layer. The values are in the form of a list of tuples, each tuple has 3 entries: (IN/OUT which is 0 or 1 detailing whehter input or output channel is affected, the layer name that is affected, offset which is the value of the offset of the filter that is affected. Offset + filter index gives the index of the filter that is affected).
        self._init_filter_dependencies()

    def _init_dropout_probs(self, drop_prob):
        """Initialize different dropout probabilities for each layer"""
        # Encoder path - each conv has its own dropout prob
        self.drop_probs["encoder1conv1"] = drop_prob
        self.drop_probs["encoder1conv2"] = drop_prob
        self.drop_probs["encoder2conv1"] = drop_prob
        self.drop_probs["encoder2conv2"] = drop_prob
        self.drop_probs["encoder3conv1"] = drop_prob
        self.drop_probs["encoder3conv2"] = drop_prob 
        self.drop_probs["encoder4conv1"] = drop_prob 
        self.drop_probs["encoder4conv2"] = drop_prob
        self.drop_probs["encoder5conv1"] = drop_prob 
        self.drop_probs["encoder5conv2"] = drop_prob 
        
        self.drop_probs["decoder5conv1"] = drop_prob
        self.drop_probs["decoder5conv2"] = drop_prob
        self.drop_probs["decoder4conv1"] = drop_prob
        self.drop_probs["decoder4conv2"] = drop_prob
        self.drop_probs["decoder3conv1"] = drop_prob
        self.drop_probs["decoder3conv2"] = drop_prob
        self.drop_probs["decoder2conv1"] = drop_prob
        self.drop_probs["decoder2conv2"] = drop_prob
        self.drop_probs["decoder1conv1"] = drop_prob
        self.drop_probs["decoder1conv2"] = drop_prob
        self.drop_probs["bottleneckconv1"] = drop_prob
        self.drop_probs["bottleneckconv2"] = drop_prob

    def _init_filter_dependencies(self):

        self.filter_dependencies["encoder1conv1"] = [(OUT, "encoder1norm1", 0), (IN, "encoder1conv2", 0)]
        self.filter_dependencies["encoder2conv1"] = [(OUT, "encoder2norm1", 0), (IN, "encoder2conv2", 0)]
        self.filter_dependencies["encoder3conv1"] = [(OUT, "encoder3norm1", 0), (IN, "encoder3conv2", 0)]
        self.filter_dependencies["encoder4conv1"] = [(OUT, "encoder4norm1", 0), (IN, "encoder4conv2", 0)]
        self.filter_dependencies["encoder5conv1"] = [(OUT, "encoder5norm1", 0), (IN, "encoder5conv2", 0)]

        self.filter_dependencies["encoder1conv2"] = [(OUT, "encoder1norm2", 0), (IN, "encoder2conv1", 0), (IN, "decoder1conv1", self.features[0])]
        self.filter_dependencies["encoder2conv2"] = [(OUT, "encoder2norm2", 0), (IN, "encoder3conv1", 0), (IN, "decoder2conv1", self.features[1])]
        self.filter_dependencies["encoder3conv2"] = [(OUT, "encoder3norm2", 0), (IN, "encoder4conv1", 0), (IN, "decoder3conv1", self.features[2])]
        self.filter_dependencies["encoder4conv2"] = [(OUT, "encoder4norm2", 0), (IN, "encoder5conv1", 0), (IN, "decoder4conv1", self.features[3])]
        self.filter_dependencies["encoder5conv2"] = [(OUT, "encoder5norm2", 0), (IN, "bottleneckconv1", 0), (IN, "decoder5conv1", self.features[4])]

        self.filter_dependencies["bottleneckconv1"] = [(OUT, "bottlenecknorm1", 0), (IN, "bottleneckconv2", 0)]
        self.filter_dependencies["bottleneckconv2"] = [(OUT, "bottlenecknorm2", 0), (OUT, "upconv5", 0)]

        self.filter_dependencies["decoder5conv1"] = [(OUT, "decoder5norm1", 0), (IN, "decoder5conv2", 0)]
        self.filter_dependencies["decoder5conv2"] = [(OUT, "decoder5norm2", 0), (OUT, "upconv4", 0)]
        self.filter_dependencies["decoder4conv1"] = [(OUT, "decoder4norm1", 0), (IN, "decoder4conv2", 0)]
        self.filter_dependencies["decoder4conv2"] = [(OUT, "decoder4norm2", 0), (OUT, "upconv3", 0)]
        self.filter_dependencies["decoder3conv1"] = [(OUT, "decoder3norm1", 0), (IN, "decoder3conv2", 0)]
        self.filter_dependencies["decoder3conv2"] = [(OUT, "decoder3norm2", 0), (OUT, "upconv2", 0)]
        self.filter_dependencies["decoder2conv1"] = [(OUT, "decoder2norm1", 0), (IN, "decoder2conv2", 0)]
        self.filter_dependencies["decoder2conv2"] = [(OUT, "decoder2norm2", 0), (OUT, "upconv1", 0)]
        self.filter_dependencies["decoder1conv1"] = [(OUT, "decoder1norm1", 0), (IN, "decoder1conv2", 0)]
        self.filter_dependencies["decoder1conv2"] = [(OUT, "decoder1norm2", 0), (IN, "final_conv", 0)]


    def forward(self, x):

        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc5))

        # Decoder path with skip connections
        dec5_up = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5_up, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4_up = self.upconv4(dec5)
        dec4 = torch.cat((dec4_up, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3_up = self.upconv3(dec4)
        dec3 = torch.cat((dec3_up, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2_up = self.upconv2(dec3)
        dec2 = torch.cat((dec2_up, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1_up = self.upconv1(dec2)
        dec1 = torch.cat((dec1_up, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output
        output = self.final_conv(dec1)
        return torch.sigmoid(output), dec1


    def _double_conv_block(self, in_channels, features, name):
        """Create a block with two convolutions, following UNet1's style with individual dropout rates"""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                            nn.Conv3d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=True,
                            )
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "drop1", nn.Dropout3d(p=self.drop_probs[name + "conv1"])),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        )
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "drop2", nn.Dropout3d(p=self.drop_probs[name + "conv2"])),
                ]
            )
        )
    

    def _single_conv_block(self, in_channels, features, name):
        """Create a block with a single convolution, following UNet1's style with individual dropout rates"""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                            nn.Conv3d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=True,
                            )
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "drop1", nn.Dropout3d(p=self.drop_probs[name + "conv1"])),
                ]
            )
        )
    

class UNet4_new(nn.Module):
    def __init__(self, in_channels=1, init_features=16, drop_prob=0.1):
        super(UNet4_new, self).__init__()
        self.drop_prob = drop_prob
        self.drop_probs = {}
        self._init_dropout_probs(drop_prob)
        features = [init_features, 2*init_features, 4*init_features, 8*init_features, 16*init_features]
        self.features = features
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.ignore_for_pruning = set(["decoder1conv2"])

        self.encoder1 = self._double_conv_block(in_channels, features[0], "encoder1")
        self.encoder2 = self._double_conv_block(features[0], features[1], "encoder2")
        self.encoder3 = self._double_conv_block(features[1], features[2], "encoder3")
        self.encoder4 = self._double_conv_block(features[2], features[3], "encoder4")
        self.bottleneck = self._double_conv_block(features[3], features[4], "bottleneck")

        # self.bottleneck = self._double_conv_block(features[4], features[5], "bottleneck")

        # self.upconv5 = nn.ConvTranspose3d(features[5], features[4], kernel_size=2, stride=2)
        # self.decoder5 = self._double_conv_block(features[4]*2, features[4], "decoder5")

        self.upconv4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = self._double_conv_block(features[3]*2, features[3], "decoder4")

        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = self._double_conv_block(features[2]*2, features[2], "decoder3")

        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._double_conv_block(features[1]*2, features[1], "decoder2")

        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._double_conv_block(features[0]*2, features[0], "decoder1")

        self.final_conv = nn.Conv3d(features[0], 1, kernel_size=1)
        self.final_conv.name = "final_conv"

        self.filter_dependencies = {} #creates a mapping from layer name to the layers that are affected when a filter is removed from the layer. The values are in the form of a list of tuples, each tuple has 3 entries: (IN/OUT which is 0 or 1 detailing whehter input or output channel is affected, the layer name that is affected, offset which is the value of the offset of the filter that is affected. Offset + filter index gives the index of the filter that is affected).
        self._init_filter_dependencies()

    def _init_dropout_probs(self, drop_prob):
        """Initialize different dropout probabilities for each layer"""
        # Encoder path - each conv has its own dropout prob
        self.drop_probs["encoder1conv1"] = drop_prob
        self.drop_probs["encoder1conv2"] = drop_prob
        self.drop_probs["encoder2conv1"] = drop_prob
        self.drop_probs["encoder2conv2"] = drop_prob
        self.drop_probs["encoder3conv1"] = drop_prob
        self.drop_probs["encoder3conv2"] = drop_prob 
        self.drop_probs["encoder4conv1"] = drop_prob 
        self.drop_probs["encoder4conv2"] = drop_prob
        # self.drop_probs["encoder5conv1"] = drop_prob 
        # self.drop_probs["encoder5conv2"] = drop_prob 
        
        # self.drop_probs["decoder5conv1"] = drop_prob
        # self.drop_probs["decoder5conv2"] = drop_prob
        self.drop_probs["decoder4conv1"] = drop_prob
        self.drop_probs["decoder4conv2"] = drop_prob
        self.drop_probs["decoder3conv1"] = drop_prob
        self.drop_probs["decoder3conv2"] = drop_prob
        self.drop_probs["decoder2conv1"] = drop_prob
        self.drop_probs["decoder2conv2"] = drop_prob
        self.drop_probs["decoder1conv1"] = drop_prob
        self.drop_probs["decoder1conv2"] = drop_prob
        self.drop_probs["bottleneckconv1"] = drop_prob
        self.drop_probs["bottleneckconv2"] = drop_prob

    def _init_filter_dependencies(self):

        self.filter_dependencies["encoder1conv1"] = [(OUT, "encoder1norm1", 0), (IN, "encoder1conv2", 0)]
        self.filter_dependencies["encoder2conv1"] = [(OUT, "encoder2norm1", 0), (IN, "encoder2conv2", 0)]
        self.filter_dependencies["encoder3conv1"] = [(OUT, "encoder3norm1", 0), (IN, "encoder3conv2", 0)]
        self.filter_dependencies["encoder4conv1"] = [(OUT, "encoder4norm1", 0), (IN, "encoder4conv2", 0)]
        # self.filter_dependencies["encoder5conv1"] = [(OUT, "encoder5norm1", 0), (IN, "encoder5conv2", 0)]

        self.filter_dependencies["encoder1conv2"] = [(OUT, "encoder1norm2", 0), (IN, "encoder2conv1", 0), (IN, "decoder1conv1", self.features[0])]
        self.filter_dependencies["encoder2conv2"] = [(OUT, "encoder2norm2", 0), (IN, "encoder3conv1", 0), (IN, "decoder2conv1", self.features[1])]
        self.filter_dependencies["encoder3conv2"] = [(OUT, "encoder3norm2", 0), (IN, "encoder4conv1", 0), (IN, "decoder3conv1", self.features[2])]
        self.filter_dependencies["encoder4conv2"] = [(OUT, "encoder4norm2", 0), (IN, "bottleneckconv1", 0), (IN, "decoder4conv1", self.features[3])]
        # self.filter_dependencies["encoder5conv2"] = [(OUT, "encoder5norm2", 0), (IN, "bottleneckconv1", 0), (IN, "decoder5conv1", self.features[4])]

        self.filter_dependencies["bottleneckconv1"] = [(OUT, "bottlenecknorm1", 0), (IN, "bottleneckconv2", 0)]
        self.filter_dependencies["bottleneckconv2"] = [(OUT, "bottlenecknorm2", 0), (OUT, "upconv4", 0)]

        # self.filter_dependencies["decoder5conv1"] = [(OUT, "decoder5norm1", 0), (IN, "decoder5conv2", 0)]
        # self.filter_dependencies["decoder5conv2"] = [(OUT, "decoder5norm2", 0), (OUT, "upconv4", 0)]
        self.filter_dependencies["decoder4conv1"] = [(OUT, "decoder4norm1", 0), (IN, "decoder4conv2", 0)]
        self.filter_dependencies["decoder4conv2"] = [(OUT, "decoder4norm2", 0), (OUT, "upconv3", 0)]
        self.filter_dependencies["decoder3conv1"] = [(OUT, "decoder3norm1", 0), (IN, "decoder3conv2", 0)]
        self.filter_dependencies["decoder3conv2"] = [(OUT, "decoder3norm2", 0), (OUT, "upconv2", 0)]
        self.filter_dependencies["decoder2conv1"] = [(OUT, "decoder2norm1", 0), (IN, "decoder2conv2", 0)]
        self.filter_dependencies["decoder2conv2"] = [(OUT, "decoder2norm2", 0), (OUT, "upconv1", 0)]
        self.filter_dependencies["decoder1conv1"] = [(OUT, "decoder1norm1", 0), (IN, "decoder1conv2", 0)]
        self.filter_dependencies["decoder1conv2"] = [(OUT, "decoder1norm2", 0), (IN, "final_conv", 0)]


    def forward(self, x):

        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        # enc5 = self.encoder5(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path with skip connections
        # dec5_up = self.upconv5(bottleneck)
        # dec5 = torch.cat((dec5_up, enc5), dim=1)
        # dec5 = self.decoder5(dec5)

        dec4_up = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4_up, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3_up = self.upconv3(dec4)
        dec3 = torch.cat((dec3_up, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2_up = self.upconv2(dec3)
        dec2 = torch.cat((dec2_up, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1_up = self.upconv1(dec2)
        dec1 = torch.cat((dec1_up, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output
        output = self.final_conv(dec1)
        return torch.sigmoid(output), dec1


    def _double_conv_block(self, in_channels, features, name):
        """Create a block with two convolutions, following UNet1's style with individual dropout rates"""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                            nn.Conv3d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=True,
                            )
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "drop1", nn.Dropout3d(p=self.drop_probs[name + "conv1"])),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        )
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "drop2", nn.Dropout3d(p=self.drop_probs[name + "conv2"])),
                ]
            )
        )
    

    def _single_conv_block(self, in_channels, features, name):
        """Create a block with a single convolution, following UNet1's style with individual dropout rates"""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                            nn.Conv3d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=True,
                            )
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "drop1", nn.Dropout3d(p=self.drop_probs[name + "conv1"])),
                ]
            )
        )