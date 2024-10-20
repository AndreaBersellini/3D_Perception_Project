import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size = 2, stride = 2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottom part
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)

    def forward(self, x):
        skip_conns = []

        for down in self.downs:
            x = down(x)
            skip_conns.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_conns = skip_conns[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_conn = skip_conns[i // 2]

            # resize in case the output is nut divisible by 2
            if x.shape != skip_conn.shape:
                x = TF.resize(x, size = skip_conn.shape[2:])

            concat_skip = torch.cat((skip_conn, x), dim = 1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)