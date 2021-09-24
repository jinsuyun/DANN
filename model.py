import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF

# class Extractor(nn.Module):
#     def __init__(self):
#         super(Extractor, self).__init__()
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
#     def forward(self, x,data=None):
#         x = self.extractor(x)  # 32 * 48 * 7 * 7
#         # a = x
#         # a = a.view(a.size(0),-1,28,28)
#         # x = x.view(-1, 3 * 28 * 28)  # 64 * 2352
#
#         return x
#
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=3 * 28 * 28, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=10),
#         )
#
#     def forward(self, x, data=None,pseudo=None):
#         x = x.view(x.size(0),-1) #Flatten
#         x = self.classifier(x)
#         if pseudo:
#             return x
#
#         return F.softmax(x)
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.discriminator = nn.Sequential(
#             nn.Linear(in_features=3 * 28 * 28, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=2),
#         )
#
#     def forward(self, input_feature, alpha, data=None):
#         input_feature = input_feature.view(input_feature.size(0), -1) #Flatten
#         reversed_input = ReverseLayerF.apply(input_feature, alpha)  # torch.Size([64, 2352])
#         x = self.discriminator(reversed_input)
#
#         return F.softmax(x)
#
# class SumDiscriminator(nn.Module):
#     def __init__(self):
#         super(SumDiscriminator, self).__init__()
#
#         self.discriminator = nn.Sequential(
#             nn.Linear(in_features=3 * 4 * 28, out_features=100),
#             # nn.Linear(in_features=3 * 28, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=2)
#         )
#
#     def forward(self, input_feature, alpha, data=None):
#         reversed_input = ReverseLayerF.apply(input_feature, alpha)
#         reversed_input = reversed_input.view(reversed_input.size(0), -1)
#
#         x = self.discriminator(reversed_input)
#         return F.softmax(x)

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

class Extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        # print(args)
        # print(kwargs['source'])

        self.source = kwargs['source']
        self.target = kwargs['target']
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=48,out_channels=1,kernel_size=5)
        )
        self.usps_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # dann model for svhn dataset
        # self.svhn_extractor1 = self.make_sequential(3, 64, 3, kernel_size=5, padding=2)
        # self.svhn_extractor2 = self.make_sequential(64, 64, 3, kernel_size=5, padding=2)
        # self.svhn_extractor3 = self.make_sequential2(64, 128, kernel_size=5, padding=2)
        self.svhn_extractor1 = self.make_sequential(3, 64, 3, kernel_size=5, padding=2)
        self.svhn_extractor2 = self.make_sequential2(64, 128, kernel_size=5, padding=2)
        self.svhn_extractor3 = self.make_sequential(128, 256, 3, kernel_size=5, padding=2)
        self.svhn_extractor4 = self.make_sequential2(256, 256, kernel_size=5, padding=2)
        self.svhn_extractor5 = self.make_sequential(256, 512, 3, kernel_size=5, padding=2)

        self.conv1x1 = nn.Conv2d(48, 1, kernel_size=1)

    def make_sequential(self, in_channels, out_channels, max_kerel_size, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(max_kerel_size))

    def make_sequential2(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, cst=False):
        # print(x.shape)
        # exit()
        if self.source == "mnist" or self.target == "mnistm":
            x = self.extractor(x)
        elif self.source == "svhn" and self.target == "mnist":
            x = self.svhn_extractor1(x)
            x = self.svhn_extractor2(x)
            x = self.svhn_extractor3(x)
            x = self.svhn_extractor4(x)
            x = self.svhn_extractor5(x)

        elif self.source == "usps" and self.target == "mnist":
            x = self.usps_extractor(x)

        if cst:
            x = self.conv1x1(x)
            return x

        # 32 * 48 * 7 * 7
        # print(source)
        # exit()
        # a = x
        # a = a.view(a.size(0),-1,28,28)
        # x = x.view(-1, 3 * 28 * 28)  # 64 * 2352

        return x


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__()
        self.source = kwargs['source']
        self.target = kwargs['target']
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )

        self.usps_classifier = nn.Sequential(
            nn.Linear(in_features=1 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )

        # self.svhn_classifier = self.make_sequential(128*3*3,3072)
        # self.svhn_classifier2 = self.make_sequential(3072,2048)
        # self.svhn_classifier3 = self.make_sequential2(2048,10)
        self.svhn_classifier = self.make_sequential(512, 256)
        self.svhn_classifier2 = self.make_sequential(256, 128)
        self.svhn_classifier3 = self.make_sequential2(128, 10)

    def make_sequential(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, *args, **kwargs),
            nn.ReLU()
        )

    def make_sequential2(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, **kwargs)

        )

    def forward(self, x, pseudo=None):
        x = x.view(x.size(0), -1)  # Flatten
        # print(x.shape) #s->m : torch.Size([32, 1152])   m->mm : torch.Size([32, 2352])

        if self.source == "mnist" or self.target == "mnistm":
            x = self.classifier(x)
        elif self.source == "svhn" and self.target == "mnist":
            x = self.svhn_classifier(x)
            x = self.svhn_classifier2(x)
            x = self.svhn_classifier3(x)

        if pseudo:
            return x

        return F.softmax(x, dim=1), x


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()
        self.source = kwargs['source']
        self.target = kwargs['target']
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
        )
        self.usps_discriminator = nn.Sequential(
            nn.Linear(in_features=1 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
        )
        # self.svhn_discriminator = self.make_sequential(128*3*3,1024)
        # self.svhn_discriminator2 = self.make_sequential(1024,1024)
        # self.svhn_discriminator3 = self.make_sequential2(1024,2)

        self.svhn_discriminator = self.make_sequential(512, 256)
        self.svhn_discriminator2 = self.make_sequential(256, 128)
        self.svhn_discriminator3 = self.make_sequential2(128, 2)

    def make_sequential(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, *args, **kwargs),
            nn.ReLU()
        )

    def make_sequential2(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, **kwargs)
        )

    def forward(self, input_feature, alpha):
        input_feature = input_feature.view(input_feature.size(0), -1)  # Flatten
        reversed_input = ReverseLayerF.apply(input_feature, alpha)  # torch.Size([64, 2352])
        if self.source == "mnist" or self.target == "mnistm":
            x = self.discriminator(reversed_input)
        elif self.source == "svhn" and self.target == "mnist":
            x = self.svhn_discriminator(reversed_input)
            x = self.svhn_discriminator2(x)
            x = self.svhn_discriminator3(x)

        return F.softmax(x, dim=1), x


class SumDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SumDiscriminator, self).__init__()
        self.source = kwargs['source']
        self.target = kwargs['target']
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 4 * 28, out_features=100),
            # nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

        self.usps_discriminator = nn.Sequential(
            nn.Linear(in_features=1 * 4 * 28, out_features=100),
            # nn.Linear(in_features=3 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

        self.svhn_discriminator = self.make_sequential(128 * 3 * 3, 1024)
        self.svhn_discriminator2 = self.make_sequential(1024, 1024)
        self.svhn_discriminator3 = self.make_sequential2(1024, 2)

    def make_sequential(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, *args, **kwargs),
            nn.ReLU()
        )

    def make_sequential2(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, **kwargs)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        reversed_input = reversed_input.view(reversed_input.size(0), -1)

        if self.source == "mnist" or self.target == "mnistm":
            x = self.discriminator(reversed_input)
        elif self.source == "svhn" and self.target == "mnist":
            x = self.svhn_discriminator(reversed_input)
            x = self.svhn_discriminator2(x)
            x = self.svhn_discriminator3(x)

        # if data == "usps" or data == "mnist":
        #     x = self.usps_discriminator(reversed_input)
        # else:
        # x = self.discriminator(reversed_input)

        return F.softmax(x, dim=1), x

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
