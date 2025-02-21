import torch
from torch import nn
from monai.networks.blocks import ResidualUnit


class FlexResNet(nn.Module):
    """
    Flexible framework to build neural network for image classification based on Residual Blocks.

    Args:
        num_classes (int): Number of output classes.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        min_size (int, optional): Minimum input size. Defaults to 32.
        initial_filters (int, optional): Number of filters in the first layer. Defaults to 64.
        num_blocks (int, optional): Number of residual blocks. Defaults to 2.
        use_global_pool (bool, optional): Whether to use global average pooling. Defaults to True.
        **residual_unit_kwargs: Additional keyword arguments for ResidualUnit.

    Raises:
        ValueError: If invalid arguments are provided.
    """
    def __init__(self, 
                num_classes, 
                in_channels=1, 
                min_size=32, 
                initial_filters=64, 
                num_blocks=2, 
                use_global_pool=True,
                **residual_unit_kwargs):
        super().__init__()
        self.min_size = min_size

        layers = []
        current_channels = in_channels

        for i in range(num_blocks):
            out_channels = initial_filters * (2**i)
            layers.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=current_channels,
                    out_channels=out_channels,
                    strides=2 if i == 0 else 1,
                    kernel_size=3,
                    subunits=2 if i == 0 else 3,
                    **residual_unit_kwargs
                )
            )
            current_channels = out_channels

        if use_global_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            layers.append(nn.Flatten())

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # Pad if necessary
        if x.size(2) < self.min_size or x.size(3) < self.min_size:
            pad_h = max(0, self.min_size - x.size(2))
            pad_w = max(0, self.min_size - x.size(3))
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    from torchsummary import summary
    model = FlexResNet(num_classes=2)
    dummy_input = torch.zeros(1, 1, 32, 32)
    model(dummy_input)
    summary(model, (1, 32, 32))
