import torch
from torchvision import datasets
import os
import torch.utils.data as data
import torchvision as tv
import hub

class FakeData(datasets.VisionDataset):
    def __init__(
        self,
        size: int = 10000,
        image_size = (1, 32, 32),
        num_classes: int = 10,
        transform = None,
        target_transform = None,
        random_offset: int = 0,
    ) -> None:
        super().__init__(None, transform=transform, target_transform=target_transform) 
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) :
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        return img, target.item()

    def __len__(self) -> int:
        return self.size


def getSVHN(batch_size, TF, data_root='/codes/twpy/arcface/data/', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='train', download=False, transform=TF), batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='test', download=False, transform=TF,), batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/codes/twpy/arcface/data/', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, download=False, transform=TF), batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, download=False, transform=TF), batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/codes/twpy/arcface/data/', TTF=None, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=True, download=False, transform=TF), batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF), batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getMNIST(batch_size, TF, data_root='/codes/twpy/arcface/data/', TTF=None, train=True, val=True, **kwargs):
    ds = []
    train_data = datasets.MNIST(root=data_root,
                                    train=True,
                                    download=False,
                                    transform = TF)

    train_loader = data.DataLoader(train_data,
                batch_size= batch_size, shuffle=True,
                drop_last=True,num_workers=2)
                
    ds.append(train_loader)

    test_data = datasets.MNIST( root=data_root,
                                    train=False,
                                    download=True,
                                    transform = TF)

    test_loader = data.DataLoader(dataset = test_data, 
                            batch_size = batch_size,
                            shuffle = False)
    ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getFashionMNIST(batch_size, TF, data_root='/codes/twpy/arcface/data/', TTF=None, train=False, val=True, **kwargs):
    data = datasets.FashionMNIST(root=data_root,
                                    download=False,
                                    train=False,
                                    transform = TF)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
   
    return loader

def getOmniglot(batch_size, TF, data_root='/codes/twpy/arcface/data/'):
    om_TF = tv.transforms.Compose([tv.transforms.ToTensor(),
                                tv.transforms.Resize((32, 32)),
                                tv.transforms.Normalize((0.1307,), (0.3081,))])
    data = datasets.Omniglot(root=data_root,
                                    download=False,
                                    transform = om_TF)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
   
    return loader


def get1dRAND(batch_size, size=2000):
    data = FakeData(size = size, image_size = (1,32,32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return loader

def get3dRAND(batch_size, size=2000):
    data = FakeData(size = size, image_size = (3,32,32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return loader


def getKvasir(batch_size, TF):
    data = datasets.ImageFolder(root='/codes/twpy/arcface/data/kvasir', transform=TF)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return loader

def getDTD(batch_size, TF):
    data = datasets.DTD(root='/codes/twpy/arcface/data/', download=False, transform=TF)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
   
    return loader

def getnotmnist(batch_size, TF):
    data = datasets.ImageFolder(root='/codes/twpy/arcface/data/notmnist', transform=TF)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader

def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=4)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'mnist':
        train_loader, test_loader = getMNIST(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=4)
    return train_loader, test_loader

def getPCAM(batch_size, TF):
    data = datasets.PCAM(root='/codes/twpy/arcface/data/', download=True, transform=TF)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
   
    return loader

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot)
    elif data_type == 'svhn':
        test_loader = getSVHN(batch_size=batch_size, train=False, TF=input_TF, data_root=dataroot)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=lambda x: 0)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == '1dfake':
        test_loader = get1dRAND(batch_size=batch_size, size=10000)
    elif data_type == '3dfake':
        test_loader = get3dRAND(batch_size=batch_size, size=10000)
    elif data_type == 'omniglot':
        test_loader = getOmniglot(batch_size=batch_size, TF=input_TF, data_root=dataroot)
    elif data_type == 'fashion':
        test_loader =getFashionMNIST(batch_size=batch_size, TF=input_TF, dataroot=dataroot)
    elif data_type == 'Kvasir':
        dataroot = '/codes/twpy/arcface/data/kvasir'
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type =='lsun':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == 'wm811k':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'wm811k'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == 'DTD':
        test_loader = getDTD(batch_size=batch_size, TF=input_TF)
    elif data_type == 'PCAM':
        test_loader = getPCAM(batch_size=batch_size, TF=input_TF)
    elif data_type == 'notmnist':
        test_loader = getnotmnist(batch_size=batch_size, TF=input_TF)
    return test_loader