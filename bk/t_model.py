# The model learns how to differentiate between an image that has cracks and that without
# The model is trained on a dataset of images with and without cracks
# The model is then used to predict the presence of cracks in new images
# The model is a convolutional neural network
# The model is trained using the Adam optimizer
# The model is trained using the binary crossentropy loss function
from pathlib import Path
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder 
from torchvision.io import read_image 
# import matplotlib.pyplot as plt
import os
import shutil
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR_PATH = "/home/program/MLDATA/data/computer_vision/cracks"
BATCH_SIZE = 8

# Define the model
class CrackDetectionModel(nn.Module):
    def __init__(self):
        super(CrackDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()  # Flatten the ouput of the pooling layer ready for the feed forwad networks
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)  # Adjust the input size
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # apply pooling to the output of the first convolution
        x = self.pool2(F.relu(self.conv2(x)))  # apply pooling to the output of the second convolution
        x = self.flatten(x)  # Use nn.Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class customDataset(Dataset):
    def __init__(self, root, data, labels, classes = None, transform=None):
        self.root = root
        self.transform = transform
        self.data = data
        self.labels = labels
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if type(image) == str:
            image = Image.open(image)
            if self.transform:
                image = self.transform(image)
        return image, label

def createCustomDataLoader(data, labels, batch_size=64, shuffle=True):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor()]
    )
    dataset = customDataset(root=DATA_DIR_PATH, data=data, labels=labels, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader

def createDataLoader(root, batch_size, shuffle = True):
    dataset = createDataset(root)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataLoader

def createDataset(root):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root, transform=transform)
    return dataset

# def visualizeImages(num_images=15, dataloader = None):
    data_loader = dataloader
    if dataloader is None:
        data_loader = createDataLoader()
    data_loader_iter = iter(data_loader)
    class_names = data_loader.dataset.classes
    images, labels = next(data_loader_iter)
    fig = plt.figure(figsize = (10, 10))
    if num_images % 5 == 0:
        n_rows = num_images // 5
    else:
        n_rows = num_images // 5 + 1
    for i in range(num_images):
        plt.subplot(n_rows, 5, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        # plt.title(class_names[i])
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

def getData(root:str):
    data = []
    labels = []
    classnames = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_path = os.path.join(root, file)
            class_name = file_path.split('/')[-2]
            label = class_name
            data.append(file_path)
            labels.append(label)
            classnames.append(class_name)
    return data, labels, classnames

def splitData(data, labels, test_size=0.2, val_size=0.2):
    # split the data into training, validation and test data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size, random_state=42)
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def trainModel(model, train_loader:DataLoader, val_loader:DataLoader, optimizer, loss_fn, epochs = 10):
    torch.manual_seed(42)
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch} .........")
        train_loss = 0
        train_accuracy = 0
        for images, labels in train_loader:
            model.train()
            output = model(images)
            labels = labels.float().view(-1, 1)

            loss = loss_fn(output, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = (output > 0.5).float()
            accuracy = accuracy_score(labels.cpu(), preds.cpu())
            train_accuracy += accuracy
        # Validate the model
        validation_loss, validation_accuracy = validate(model, val_loader, loss_fn)
        print(f"Epoch: {epoch} Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

def validate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            output = model(images)
            labels = labels.float().view(-1, 1)
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            
            preds = (output > 0.5).float()
            accuracy = accuracy_score(labels.cpu(), preds.cpu())
            total_accuracy += accuracy
    
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    return avg_loss, avg_accuracy

def splitFiles(root, label1_name, label2_name, test_split = 0.2, val_split = 0.2):
    label1_files = []
    label2_files = []
    train_dir = os.path.join(root, 'train')
    train_label1_dir = os.path.join(train_dir, label1_name)
    train_label2_dir = os.path.join(train_dir, label2_name)
    test_dir = os.path.join(root,"test")
    test_label1_dir = os.path.join(test_dir, label1_name)
    test_label2_dir = os.path.join(test_dir, label2_name)
    val_dir = os.path.join(root, 'val')
    val_label1_dir = os.path.join(val_dir, label1_name)
    val_label2_dir = os.path.join(val_dir, label2_name)
    __dirs = [train_dir, train_label1_dir, train_label2_dir, test_dir, test_label1_dir, test_label2_dir, val_dir, val_label1_dir, val_label2_dir]

    for __dir in __dirs:
        if not Path(__dir).exists():
            os.mkdir(__dir)
    label1_dir = os.path.join(root, label1_name)
    label2_dir = os.path.join(root, label2_name)
    if  Path(label1_dir).exists():
        dirs = [label1_dir, label2_dir]
        for __dir in dirs:
            for root, dirs, files in os.walk(__dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if __dir == label1_dir:
                        label1_files.append(file_path)
                    else:
                        label2_files.append(file_path)
    
        print(f"n_label1_files: {len(label1_files)}")
        print(f"n_label2_files: {len(label2_files)}")
        # split the label files into traing, testing, validation files and move them to the respecitve folders
        train_label1_files, test_label1_files = train_test_split(label1_files, test_size=test_split)
        train_label1_files, val_label1_files = train_test_split(train_label1_files, test_size=val_split)
        train_label2_files, test_label2_files = train_test_split(label2_files, test_size=test_split)
        train_label2_files, val_label2_files = train_test_split(train_label2_files, test_size=val_split)

        print(f"num_train_label1files: {len(train_label1_files)}")
        print(f"num_test_label1files: {len(test_label1_files)}")
        print(f"num_val_label1files: {len(val_label1_files)}")
        # move the files to the respective folders

        for file in train_label1_files:
            shutil.move(file, os.path.join(train_dir, label1_name, file.split('/')[-1]))
        for file in train_label2_files:
            shutil.move(file, os.path.join(train_dir, label2_name, file.split('/')[-1]))
        for file in test_label1_files:
            shutil.move(file, os.path.join(test_dir, label1_name, file.split('/')[-1]))
        for file in test_label2_files:
            shutil.move(file, os.path.join(test_dir, label2_name, file.split('/')[-1]))        
        for file in val_label1_files:
            shutil.move(file, os.path.join(val_dir, label1_name, file.split('/')[-1]))
        for file in val_label2_files:
            shutil.move(file, os.path.join(val_dir, label2_name, file.split('/')[-1]))

        os.rmdir(label1_dir)
        os.rmdir(label2_dir)
        return train_dir, test_dir, val_dir 
    else:
        return train_dir, test_dir, val_dir

# device agnostic code
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

train_dir, test_dir, val_dir = splitFiles(DATA_DIR_PATH, "Negative", "Positive")
train_dataloader = createDataLoader(train_dir, batch_size=BATCH_SIZE)
test_dataloader = createDataLoader(test_dir, batch_size=BATCH_SIZE)
val_dataloader = createDataLoader(val_dir, batch_size=BATCH_SIZE)

# create the model, optimizer and loss function
model  = CrackDetectionModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# train the model
trainModel(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs= 10)

# test the model using the test DataLoader
test_loss, test_accuracy = validate(model, test_dataloader, loss_fn)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
