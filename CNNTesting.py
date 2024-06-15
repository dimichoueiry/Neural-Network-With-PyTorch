[] # Step 1: Importing the necessary libraries
import torch  # this is for tensor computation
import torch.nn as nn  # Neural Network
import torch.optim as optim  # Optimization
import torchvision  # this is for image processing
from torchvision import datasets, models, transforms  # this is for image processing
from sklearn.model_selection import train_test_split  # for splitting the dataset
from torch.utils.data import Subset  # for creating dataset subsets
import random  # this is for random number generation
import numpy as np  # this is for numerical python
import matplotlib.pyplot as plt  # this is for plotting

[] # Step 2: Setting the seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

[] # Step 3: Loading the dataset
# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset. Dataset is divided into training, validation and testing datasets
full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


# First split: 85% train + validation, 15% test
train_valid_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size=0.15, random_state=42)

# Second split: 70% train, 15% validation
train_indices, valid_indices = train_test_split(train_valid_indices, test_size=0.1765, random_state=42)  # 0.1765 ≈ 15 / 85

# Create the subsets
train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)
test_dataset = Subset(full_dataset, test_indices)

# Define the dataloaders. Data loaders are used to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

[] # Step 4: Defining the model
# Define the model.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the convolutional, activation function pooling, and fully connected layers of the model

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # Define the activation function
        self.Leaky = nn.LeakyReLU(0.1)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Define the forward pass
        x = self.pool(self.Leaky(self.conv1(x)))
        x = self.pool(self.Leaky(self.conv2(x)))
        x = self.pool(self.Leaky(self.conv3(x)))

        # Flatten the output
        x = x.view(-1, 128 * 28 * 28)

        # Pass the output through the fully connected layers
        x = self.Leaky(self.fc1(x))
        x = self.fc2(x)

        return x

        
# Create an instance of the model
model = CNN().to(device)

[] # Step 5: Defining the loss function and optimizer
# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

[] # Step 6: Training the model 
# Define the number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.item()

        if i % 100 == 99:
            print('[Epoch %d, Step %5d] running loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # Compute average training loss over the epoch
    train_loss /= len(train_loader)

    # Compute validation loss
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    validation_loss /= len(valid_loader)

    print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.3f}, Validation Loss: {validation_loss:.3f}')

print('Finished Training')

[] # Step 7: Evaluating the model
# Set the model to evaluation mode
model.eval()

# Initialize the number of correct predictions to 0
correct = 0

# Initialize the total number of predictions to 0
total = 0

# Initialize the confusion matrix
confusion_matrix = np.zeros((10, 10))

# Do not calculate the gradients because we are not training
with torch.no_grad():
    for data in test_loader:
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Get the outputs
        outputs = model(inputs)

        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)

        # Update the total number of predictions
        total += labels.size(0)

        # Update the number of correct predictions
        correct += (predicted == labels).sum().item()

        # Update the confusion matrix
        for i in range(len(labels)):
            confusion_matrix[labels[i]][predicted[i]] += 1

# Print the accuracy
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Print the confusion matrix
print('Confusion matrix:')

# Print the column headers
print('      ', end='')

for i in range(10):
    print('{0:5}'.format(classes[i]), end='')
print()

# Print the rows
for i in range(10):
    print('{0:5}'.format(classes[i]), end='')
    for j in range(10):
        print('{0:5}'.format(int(confusion_matrix[i][j])), end='')
    print()


[] # Step 8: Save the model to make predictions
PATH = 'model.pth'
torch.save(model.state_dict(), PATH)

[] # Step 9: Load the model to make predictions
model = CNN()
model.load_state_dict(torch.load(PATH))
model.to(device)


[] # Step 10: Making predictions
dataiter = iter(test_loader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# print images
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))

