from torchvision import datasets #Import datasets module from torchvision to access MNIST dataset
from torchvision.transforms import ToTensor #Import ToTensor transform to convert images into PyTorch tensors

#Load MNIST training dataset
train_data = datasets.MNIST ( 
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

#Load MNIST test dataset
test_data = datasets.MNIST ( 
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)


###
#Calls for train_data and test_data to view basic information about the datasets

#View number of data points, root location, split (train/test), and applied transform
train_data           
test_data            

#View shape of the data tensors in train_data and test_data
train_data.data.shape  # Displays shape of training data tensor ([60000, 28, 28] for 60,000 images of 28x28 pixels)
test_data.data.shape   # Displays shape of test data tensor ([10000, 28, 28] for 10,000 images of 28x28 pixels)

#View size of target labels for train_data
train_data.targets.size()

#View the actual target labels for train_data
train_data.targets
###


from torch.utils.data import DataLoader #Import DataLoader to manage batches of data for training and testing

#Create data loaders for training and testing datasets
loaders = {

    'train': DataLoader(train_data, 
                        batch_size=100,  #Number of samples per batch
                        shuffle=True,    #Shuffle the data
                        num_workers=1),  #Number of worker threads

    'test': DataLoader(test_data,
                       batch_size=100,
                       shuffle=True,
                       num_workers=1),
}

#Output train and test DataLoader objects
loaders