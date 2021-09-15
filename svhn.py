import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params

# train_transform = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize(mean=(0.45141874380092256,),
#                                                            std=(0.19929124669110937,)),
#                                       transforms.Resize((28, 28))
#                                       ])
# test_transform = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize(mean=(0.4579653771401513, 0.5, 0.5),
#                                                           std=(0.2250053592015834,)),
#                                      transforms.Resize((28, 28))
#                                      ])

train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.437,0.4437,0.4728),
                                                           std=(0.1980,0.2010,0.1970)),
                                      transforms.Resize((28, 28))
                                      ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.437, 0.4437, 0.4728),
                                                          std=(0.1980,0.2010,0.1970)),
                                     transforms.Resize((28, 28))
                                     ])


svhn_train_dataset = datasets.SVHN(root='data/pytorch/SVHN', split="train", download=True,
                                   transform=train_transform)
svhn_valid_dataset = datasets.SVHN(root='data/pytorch/SVHN', split="train", download=True,
                                   transform=transforms)
svhn_test_dataset = datasets.SVHN(root='data/pytorch/SVHN', split="test", download=True,
                                  transform=test_transform)

# print('==> SVHN Computing mean and std..')
# mean_train = svhn_train_dataset.data.mean()
# mean_val = svhn_valid_dataset.data.mean()
# mean_test = svhn_test_dataset.data.mean()
#
# std_train = svhn_train_dataset.data.std()
# std_val = svhn_valid_dataset.data.std()
# std_test = svhn_test_dataset.data.std()
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

indices = list(range(len(svhn_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

svhn_train_loader = DataLoader(
    svhn_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

svhn_valid_loader = DataLoader(
    svhn_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

svhn_test_loader = DataLoader(
    svhn_test_dataset,
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
