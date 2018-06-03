import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def create_dataloader(config, dataset=None):
    if dataset is not None:
        return DataLoader(dataset=dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=config.num_workers)
    
    if config.mnist_path is None:
        dataset = MNIST(root='.', transform=transforms.ToTensor(), download=True)
    else:
        dataset = MNIST(root=config.mnist_path, transform=transforms.ToTensor())
        
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
    return dataloader