# Import necessary libraries
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Function to download and load datasets
def load_data(batch_size=256):
    """Loads the MNIST dataset and returns data loaders for training and testing."""
    transform = transforms.ToTensor()
    trainset = datasets.MNIST('train', download=True, train=True, transform=transform)
    testset = datasets.MNIST('test', download=True, train=False, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader



# Define the CNN model
class MyModel(nn.Module):
    def __init__(self):
        """Initializes the CNN model layers."""
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*24*24, 512)  # Adjusted size
        self.fc2 = nn.Linear(512, 128)       # Added another fully connected layer
        self.fc3 = nn.Linear(128, 10)

    def forward(self, images):
        """Forward pass through the network."""
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(model, trainloader, epochs=15, lr=0.0005):
    """Trains the CNN model and returns the training loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    train_loss = []

    for epoch in range(epochs):
        epoch_loss = []
        model.train()  # Set the model to training mode
        for images, labels in trainloader:
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return train_loss

# Function to test the model and calculate accuracy
def test_model(model, testloader):
    """Tests the CNN model on the test dataset and returns accuracy."""
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation during testing
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy

# Function to plot training loss curve
def plot_loss(train_loss):
    """Plots the training loss curve."""
    plt.plot(train_loss, label='Training Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Main function to run the entire process
def main():
    # Load data
    trainloader, testloader = load_data()

    

    # Initialize the model
    model = MyModel()

    # Train the model
    print("Training the model...")
    epochs = 15
    train_loss = train_model(model, trainloader, epochs=epochs, lr=0.0005)

    # Plot training loss
    plot_loss(train_loss)

    # Test the model and show accuracy
    print("Evaluating the model on the test dataset...")
    test_model(model, testloader)

    # Save the model
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print("Model saved as 'mnist_cnn_model.pth'")

if __name__ == "__main__":
    main()
