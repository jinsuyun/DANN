import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)  # 32 * 48 * 7 * 7
        # a = x
        # a = a.view(a.size(0),-1,28,28)
        # x = x.view(-1, 3 * 28 * 28)  # 64 * 2352

        return x


'''Separate extractor 1, 2'''


# class Extractor(nn.Module):
#     def __init__(self):
#         super(Extractor, self).__init__()
#         self.extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.extractor2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#
#     def forward(self, x):
#         x = self.extractor(x)  # 32 * 48 * 7 * 7
#         extractor = x
#
#         x = self.extractor2(x)
#         extractor2 = x
#         x = x.view(-1, 3 * 28 * 28)  # 64 * 2352
#
#         return x, extractor, extractor2


# class ExtractorFeatureMap(nn.Module):
#     def __init__(self):
#         super(ExtractorFeatureMap, self).__init__()
#         self.extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#
#             nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#
#     def forward(self, x):
#         x = self.extractor(x)
#
#         # x = x.view(-1, 3, 28 ,28)
#         return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )

    def forward(self, x, pseudo=None):
        x = x.view(x.size(0),-1) #Flatten
        x = self.classifier(x)
        if pseudo:
            return x

        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
        )

    def forward(self, input_feature, alpha):
        input_feature = input_feature.view(input_feature.size(0), -1) #Flatten
        reversed_input = ReverseLayerF.apply(input_feature, alpha)  # torch.Size([64, 2352])
        x = self.discriminator(reversed_input)

        return F.softmax(x)


class SumDiscriminator(nn.Module):
    def __init__(self):
        super(SumDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 4 * 28, out_features=100),
            # nn.Linear(in_features=3 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        reversed_input = reversed_input.view(reversed_input.size(0), -1)

        x = self.discriminator(reversed_input)
        return F.softmax(x)

# class Discriminator_Conv2d(nn.Module):
#     def __init__(self):
#         super(Discriminator_Conv2d, self).__init__()
#         self.discriminator = nn.Sequential(
#             nn.Conv2d(3*4*28, 512, kernel_size=1, stride=1, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
#         )
#
#     def forward(self, x,alpha):
#         reversed_input = ReverseLayerF.apply(x,alpha)
#         reversed_input = reversed_input.view(reversed_input.size(0), -1)
#         x = self.discriminator(reversed_input)
#         return F.softmax(x)

# class _InstanceDA(nn.Module):
#     def __init__(self, in_channel=4096):
#         super(_InstanceDA, self).__init__()
#         self.dc_ip1 = nn.Linear(in_channel, 1024)
#         self.dc_relu1 = nn.ReLU()
#         self.dc_drop1 = nn.Dropout(p=0.5)
#
#         self.dc_ip2 = nn.Linear(1024, 1024)
#         self.dc_relu2 = nn.ReLU()
#         self.dc_drop2 = nn.Dropout(p=0.5)
#
#         self.clssifer = nn.Linear(1024, 1)
#         self.LabelResizeLayer = InstanceLabelResizeLayer()
#
#     def forward(self, x, need_backprop):
#         x = grad_reverse(x)
#         x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
#         x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
#         x = F.sigmoid(self.clssifer(x))
#         label = self.LabelResizeLayer(x, need_backprop)
#         return x, label
