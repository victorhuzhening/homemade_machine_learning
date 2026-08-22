import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d(nn.Module):
    """
    Custom 2D Convolutional class equipped with tuneable knobs 
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation=None, weights_init='xavier_uniform'):
        super(conv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weights_init = weights_init
        self.activation = activation

        self.weights = nn.Parameter(
            torch.empty(out_channel, in_channel, kernel_size, kernel_size, dtype=torch.float32)
        )

        self.bias = nn.Parameter(
            torch.empty(out_channel, dtype=torch.float32)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)

            if self.weights_init == 'xavier_uniform':
                nn.init.xavier_uniform_(self.weights)
            elif self.weights_init == 'xavier_normal':
                nn.init.xavier_normal(self.weights)
            elif self.weights_init == 'kaiming_uniform':
                nn.init.kaiming_uniform(self.weights)
            elif self.weights_init == 'kaiming_normal':
                nn.init.kaiming_normal(self.weights)

    def forward(self, input):
        if self.activation == 'relu':
            return F.relu((F.conv2d(input, self.weights, self.bias, self.stride, self.padding)), inplace=True)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu((F.conv2d(input, self.weights, self.bias, self.stride, self.padding)))
        
        return F.conv2d(input, self.weights, self.bias, self.stride, self.padding)



class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.image_features = nn.Sequential(
            self._conv_block(3, 64, 2),
            self._conv_block(64, 128, 2),
            self._conv_block(128, 256, 2),
            self._conv_block(256, 512, 3),
            self._conv_block(512, 512, 3)
        )

        self.adaptive_layer = nn.AdaptiveMaxPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _conv_block(self, in_channel, out_channel, num_convs, kernel_size=3, stride=1, padding=1, activation='relu'):
        layers = nn.Sequential()
        for i in range(num_convs):
            layers.append(
                conv2d(in_channel=in_channel if i == 0 else out_channel,
                       out_channel=out_channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       activation=activation
                )
            )
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers


    def forward(self, x):
        x = self.image_features(x)
        x = self.adaptive_layer(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    
