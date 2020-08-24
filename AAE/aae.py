import torch
import torch.nn as nn


def conv_bn(inp, oup, ks=5, stride=2, padding=2):
    return nn.Sequential(
        nn.Conv2d(inp, oup, ks, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def upsample_conv_bn(inp, oup, ks=5, stride=1, padding=2, scale_factor=2, mode='nearest', is_Sigmoid=False):
    if is_Sigmoid:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(inp, oup, ks, stride, padding, bias=False),
            nn.BatchNorm2d(oup),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(inp, oup, ks, stride, padding, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )


class AugmentedAutoEncoder(nn.Module):
    def __init__(self, implict_dim=128, input_size=128, obj_nums=79):
        super(AugmentedAutoEncoder, self).__init__()

        last_size = input_size
        encoder_channels = [3, 128, 256, 256, 512]
        decoder_channels = encoder_channels[::-1]
        self.implict_dim = implict_dim

        self.encoder = []
        for i in range(len(encoder_channels) - 1):
            inp = encoder_channels[i]
            oup = encoder_channels[i + 1]
            self.encoder.append(conv_bn(inp, oup))
            last_size = last_size / 2
        last_size *= 2  # 多除了一次
        self.encoder = nn.Sequential(*self.encoder)
        self.fc1 = nn.Linear(int(last_size * last_size * encoder_channels[-1]), implict_dim)

        self.decoder = []
        # the last layer's activative function is different.
        for i in range(len(decoder_channels) - 1):
            inp = decoder_channels[i]
            oup = decoder_channels[i + 1]
            if i == len(decoder_channels) - 2:  # 最后一层用sigmoid
                self.decoder.append(
                    upsample_conv_bn(inp, oup, is_Sigmoid=True)
                )
            else:
                self.decoder.append(
                    upsample_conv_bn(inp, oup, is_Sigmoid=False)
                )

        self.decoder = nn.Sequential(*self.decoder)
        self.fc2 = nn.Linear(implict_dim, int(last_size * last_size * encoder_channels[-1]))

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        b, c, h, w = x.size()
        x = x.view(-1, c * h * w)
        # latent vector
        x = self.fc1(x)

        # reconstruction
        x = self.fc2(x)
        x = x.view(-1, c, h, w)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    aae = AugmentedAutoEncoder()
    input_size = (1, 3, 256, 256)
    input_tensor = torch.randn(input_size, dtype=torch.float)
    output_x, pred_cls = aae(input_tensor)
    import pdb;

    pdb.set_trace()
