from torchvision import datasets                  #Import datasets module from torchvision to access MNIST dataset
from torchvision.transforms import ToTensor       #Import ToTensor transform to convert images into PyTorch tensors

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
train_data.data.shape  #Display shape of training data tensor
test_data.data.shape   #Display shape of test data tensor

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


import torch.nn as nn               #Import PyTorch neural network module to define and create layers
import torch.nn.functional as F     #Import functional API for operations
import torch.optim as optim         #Import optimization algorithms

#Define Convolutional Neural Network class
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__() #Initialize parent class

        # Define 1st convolutional layer
        # Takes 1 input channel (grayscale images) and outputs 10 feature maps
        # Uses a 5x5 kernel (filter) for convolution
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # Define 2nd convolutional layer
        # Takes 10 input channels (from previous layer) and outputs 20 feature maps        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Add a dropout layer to reduce overfitting by randomly zeroing some channels
        self.conv2_drop = nn.Dropout2d()

        # Define 1st fully connected layer
        # Maps 320 input features (flattened feature maps) to 50 output features
        self.fc1 = nn.Linear(320, 50)

        # Define 2nd fully connected layer
        # Maps 50 input features to 10 output classes (digits 0-9 for classification)   
        self.fc2 = nn.Linear(50, 10)

    #Define forward pass of the CNN
    def forward(self, x):
        
        # Apply 1st convolutional layer followed by ReLU activation and max pooling
        # Max pooling reduces spatial dimensions by a factor of 2 (2x2 pooling window)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Apply the 2nd convolutional layer followed by dropout, ReLU activation, and max pooling
        # Dropout helps prevent overfitting by randomly zeroing some feature maps during training
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Flatten the feature maps into a 1D vector for input to the fc layers
        # -1 allows PyTorch to infer batch size automatically
        x = x.view(-1, 320)
        
        # 1st fc layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        
        # Dropout to the output of the 1st fc layer during training
        x = F.dropout(x, training=self.training)
        
        # 2nd fc layer (output layer)
        x = self.fc2(x)
        
        # softmax function to produce probabilities for each class
        return F.softmax(x, dim=1)