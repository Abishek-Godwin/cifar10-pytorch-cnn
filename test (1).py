import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision 
import torchvision.transforms as transforms

def compute_mean_std():
    temp_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='D:/data', train=True, download=True, transform=temp_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()

def main():
    # Step 1: Compute mean and std
    mean, std = compute_mean_std()
    

    # Step 2: Use the computed values in transform
    transform_images = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    test_data = torchvision.datasets.CIFAR10(
        root='D:/data', train=False, download=True, transform=transform_images)

    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False, num_workers=2)

    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  
    class CNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 36, 4,padding = 1) 
            self.pool = nn.MaxPool2d(2, 2) 
            self.conv2 = nn.Conv2d(36, 36, 4)
            self.conv3 = nn.Conv2d(36, 76,3)
            self.conv4 = nn.Conv2d(76,130,3)
            self.fc1 = nn.Linear(130 * 5 * 5, 400)
            self.fc2 = nn.Linear(400, 84)
            self.fc3 = nn.Linear(84, 10)
            self.dropout = nn.Dropout(0.5) 
            self.dropout2 = nn.Dropout2d(0.4)
            self.bn1 = nn.BatchNorm2d(36)
            self.bn2 = nn.BatchNorm2d(36)
            self.bn3 = nn.BatchNorm2d(76)
            self.bn4 = nn.BatchNorm2d(130)
            self.bn5 = nn.BatchNorm1d(400)
            self.bn6 = nn.BatchNorm1d(84)



        def forward(self, x):
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = self.pool(self.dropout2(F.leaky_relu(self.bn4(self.conv4(x)))))
            x = torch.flatten(x,1) # flatten the tensor from 2D to 1D
            x = self.dropout(F.leaky_relu(self.bn5(self.fc1(x))))
            x = F.leaky_relu(self.bn6(self.fc2(x)))
            x = self.fc3(x)
            return x 


    net = CNN()

    net.load_state_dict(torch.load('trained_cnn.pth'))

    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

if __name__ == "__main__":
    main()