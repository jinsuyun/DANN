import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params

# train_transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=(0.2468769476570631,),
#                                                      std=(0.29887581181519896,))
#                                 ])
# test_transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=(0.2598811329464521,),
#                                                      std=(0.30825718086426723,))
#                                 ])
train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.2539,),
                                                           std=(0.3842,)),
                                      transforms.Resize((28, 28))
                                      ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.2539,),
                                                          std=(0.3842,)),
                                     transforms.Resize((28, 28))
                                     ])

usps_train_dataset = datasets.USPS(root='data/pytorch/USPS', train=True, download=True,
                                   transform=train_transform)
usps_valid_dataset = datasets.USPS(root='data/pytorch/USPS', train=True, download=True,
                                   transform=transforms)
usps_test_dataset = datasets.USPS(root='data/pytorch/USPS', train=False, download=True,
                                  transform=test_transform)

# print('==> Computing mean and std..')
# mean_train = usps_train_dataset.data.mean(axis=(0,1,2))
# mean_val = usps_valid_dataset.data.mean(axis=(0,1,2))
# mean_test = usps_test_dataset.data.mean(axis=(0,1,2))
#
# std_train = usps_train_dataset.data.std(axis=(0,1,2))
# std_val = usps_valid_dataset.data.std(axis=(0,1,2))
# std_test = usps_test_dataset.data.std(axis=(0,1,2))
#
# mean_train = mean_train / 255
# mean_val = mean_val / 255
# mean_test = mean_test / 255
#
# std_train = std_train / 255
# std_val = std_val / 255
# std_test = std_test / 255
# print("Mean")
# print(mean_train, mean_val,mean_test)
#
# print("Std")
# print(std_train, std_val,std_test)
# print("\n")

indices = list(range(len(usps_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

usps_train_loader = DataLoader(
    usps_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

usps_valid_loader = DataLoader(
    usps_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

usps_test_loader = DataLoader(
    usps_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

# print(one_hot_embedding(mnist_test_dataset.test_labels))

# print(mnist_concat.shape)
