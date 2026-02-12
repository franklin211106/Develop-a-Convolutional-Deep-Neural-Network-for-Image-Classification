## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.
### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.
### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.
### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM
### Name:FRANKLIN F
### Register Number:212224240041

```

   class CNNClassifier(nn.Module):
    def __init__(self):
       super(CNNClassifier, self).__init__()
       self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
       self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
       self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
       self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
       self.fc1=nn.Linear(128*3*3,128)
       self.fc2=nn.Linear(128,64)
       self.fc3=nn.Linear(64,10)
    def forward(self,x):
       x=self.pool(torch.relu(self.conv1(x)))
       x=self.pool(torch.relu(self.conv2(x)))
       x=self.pool(torch.relu(self.conv3(x)))
       x=x.view(x.size(0),-1)
       x=torch.relu(self.fc1(x))
       x=torch.relu(self.fc2(x))
       x=self.fc3(x)
       return x


```

### OUTPUT

## Training Loss per Epoch
<img width="374" height="164" alt="image" src="https://github.com/user-attachments/assets/c22f7151-71ac-4f82-b44b-4f49e1de3c76" />



## Confusion Matrix
<img width="780" height="630" alt="image" src="https://github.com/user-attachments/assets/524f6284-2fcf-4567-8596-2b14838b231b" />

<img width="837" height="533" alt="image" src="https://github.com/user-attachments/assets/83a1c8bd-c3ec-4046-85fe-2981e72e908d" />

## Classification Report
<img width="550" height="362" alt="image" src="https://github.com/user-attachments/assets/bc523087-9b3b-415a-9c0a-7cb43697c549" />


### New Sample Data Prediction
<img width="613" height="478" alt="image" src="https://github.com/user-attachments/assets/73856b0b-6a8e-4ab9-befe-999b74dd9929" />



## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
