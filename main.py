from torchvision import datasets #Import datasets module from torchvision to access MNIST dataset
from torchvision.transforms import ToTensor #Import ToTensor transform to convert images into PyTorch tensors

#Load MNIST training dataset
train_data = datasets.MNIST( 
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

#Load MNIST test dataset
test_data = datasets.MNIST( 
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)