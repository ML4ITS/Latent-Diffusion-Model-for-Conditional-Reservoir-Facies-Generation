import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, noise):
        x = self.model(x)
        x = torch.cat((x, skip_input, noise), 1)

        return x


class UNetUpNoise(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_size: int):
        super(GeneratorUNet, self).__init__()
        self.img_size = img_size

        # downsampling layers
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # upsampling layers
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(512*3, 512, dropout=0.5)
        self.up3 = UNetUp(512*3, 512, dropout=0.5)
        self.up4 = UNetUp(512*3, 512, dropout=0.5)
        self.up5 = UNetUp(512*3, 256)
        self.up6 = UNetUp(256*3, 128)
        self.up7 = UNetUp(128*3, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64*3, out_channels, 4, padding=1),
            # nn.Tanh(),
        )

        # upsampling layers for a random vector
        self.up1_n = UNetUpNoise(512, 512)
        self.up2_n = UNetUpNoise(512, 512)
        self.up3_n = UNetUpNoise(512, 512)
        self.up4_n = UNetUpNoise(512, 512)
        self.up5_n = UNetUpNoise(512, 256)
        self.up6_n = UNetUpNoise(256, 128)
        self.up7_n = UNetUpNoise(128, 64)

    def forward(self, x, return_noise_vector=False):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        noise = torch.randn_like(d8).to(x.device)
        u1_n = self.up1_n(noise)
        u2_n = self.up2_n(u1_n)
        u3_n = self.up3_n(u2_n)
        u4_n = self.up4_n(u3_n)
        u5_n = self.up5_n(u4_n)
        u6_n = self.up6_n(u5_n)
        u7_n = self.up7_n(u6_n)

        # upsampling layers
        u1 = self.up1(d8, d7, u1_n)
        u2 = self.up2(u1, d6, u2_n)
        u3 = self.up3(u2, d5, u3_n)
        u4 = self.up4(u3, d4, u4_n)
        u5 = self.up5(u4, d3, u5_n)
        u6 = self.up6(u5, d2, u6_n)
        u7 = self.up7(u6, d1, u7_n)

        out = self.final(u7)

        if return_noise_vector:
            return out, noise
        else:
            return out

    def scale_to_original(self, out):
        out = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear')
        return out


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        """
        img_A: fake_x (generated x)
        img_B: x_cond
        """
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        out = self.model(img_input)
        return out


if __name__ == '__main__':
    b = 1
    n_categories = 4
    img_size = 128
    unet_in_size = 256

    x = torch.rand((b, n_categories, unet_in_size, unet_in_size))
    x_cond = torch.rand((b, n_categories+1, unet_in_size, unet_in_size))

    generator = GeneratorUNet(n_categories + 1, n_categories, img_size)
    discriminator = Discriminator(1)

    out_gen = generator(x_cond)
    print('out_gen.shape:', out_gen.shape)

    out_disc = discriminator(torch.argmax(x, dim=1, keepdim=True).float(),
                             torch.argmax(out_gen, dim=1, keepdim=True).float())
    print('out_disc.shape:', out_disc.shape)
