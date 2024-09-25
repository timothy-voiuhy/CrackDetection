import torch
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torch.optim as optim 
from torchvision.datasets import ImageFolder 
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from pathlib import Path
import sys
import shutil
import matplotlib.pyplot as plt
import argparse

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

def createCustomDataLoader(root, data, labels, batch_size=64, shuffle=True):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor()]
    )
    dataset = customDataset(root=root, data=data, labels=labels, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader


class CrackDetectionModel(nn.Module):
    def __init__(self):
        super(CrackDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) #64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) #128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()  # Flatten the ouput of the pooling layer ready for the feed forwad networks
        self.fc1 = nn.Linear(32 * 32 * 32, 1024)  # Adjust the input size note the first integer here is the number of channels that are in the last convultional neural network
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

def createDataLoader(root, batch_size, shuffle = True):
    dataset = createDataset(root)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader

def createDataset(root):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root, transform=transform)
    return dataset

def trainModel(model:CrackDetectionModel, train_loader:DataLoader, val_loader:DataLoader, optimizer, loss_fn, epochs = 10, device = None, save_path = None, batch_size = 64):
    torch.manual_seed(42)
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch} .........")
        train_loss = 0
        train_accuracy = 0
        work_image_count  = 0
        for batch_index, (images, labels) in  enumerate(train_loader):
            model.train()
            if device:
                images = images.to(device)
                labels = labels.to(device)
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
            if (batch_index+1) % 10 == 0:
                print(f"Worked on : {batch_index + 1} of {len(train_loader)} batches")
            work_image_count += len(images)
            if work_image_count % 1000 == 0:
                print(f"Worked on : {work_image_count} images / {len(train_loader)*batch_size} images")

        # Validate the model per epoch count
        validation_loss, validation_accuracy = validate(model, val_loader, loss_fn, device=device)
        average_loss = train_loss / len(train_loader)
        average_accuracy = train_accuracy /len(train_loader)
        print(f"Epoch: {epoch} Train Loss: {average_loss}, Train Accuracy: {average_accuracy}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")
        if validation_loss < best_val_loss:
            print("Saving model .............")
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

def validate(model, data_loader, loss_fn, device = None):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
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
    if  Path(label1_dir).exists(): # if one of the label directories exists then the function has not been run.
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
        delimeter = ""
        if sys.platform == "win32":
            delimeter = "\\"
        elif sys.platform == "linux":
            delimeter = "/"
        for file in train_label1_files:
            shutil.move(file, os.path.join(train_dir, label1_name, file.split(delimeter)[-1]))
        for file in train_label2_files:
            shutil.move(file, os.path.join(train_dir, label2_name, file.split(delimeter)[-1]))
        for file in test_label1_files:
            shutil.move(file, os.path.join(test_dir, label1_name, file.split(delimeter)[-1]))
        for file in test_label2_files:
            shutil.move(file, os.path.join(test_dir, label2_name, file.split(delimeter)[-1]))
        for file in val_label1_files:
            shutil.move(file, os.path.join(val_dir, label1_name, file.split(delimeter)[-1]))
        for file in val_label2_files:
            shutil.move(file, os.path.join(val_dir, label2_name, file.split(delimeter)[-1]))

        os.rmdir(label1_dir)
        os.rmdir(label2_dir)
        return train_dir, test_dir, val_dir
    else:
        return train_dir, test_dir, val_dir

def visualizeImages(num_images=15, root = None, batch_size = 64):
    data_loader = createDataLoader(root, batch_size)
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

def splitData(data, labels, test_size=0.2, val_size=0.2):
    # split the data into training, validation and test data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size, random_state=42)
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def predict_single_image(image_path, model=None, device=None, model_path=None):
    """Predicts the class of a single crack image.

    Args:
        image_path: Path to the image file.
        model: The trained PyTorch model.
        device: The device to run the prediction on (CPU or GPU).

    Returns:
        The predicted class (0 for no crack, 1 for crack).
    """
    if model_path is not None:
        model = torch.load(model_path)
    else:
        model = model
    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the device
    if device:
        image = image.to(device)

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)

    # Get the predicted class
    predicted_class = (output > 0.5).float().item()  # Assuming binary classification

    return predicted_class

def testModel(model, data_loader, loss_fn, device = None):
  model.eval()
  total_loss = 0
  total_accuracy = 0
  with torch.inference_mode():
    for images, labels in data_loader:
      if device is not None:
        images = images.to(device)
        labels = labels.to(device)
      output = model(images)
      labels = labels.float().view(-1, 1)
      loss = loss_fn(output, labels)
      total_loss += loss.item() # accumulatively calculate the loss
      preds = (output > 0.5).float()
      accuracy = accuracy_score(labels.cpu(), preds.cpu())
      total_accuracy += accuracy # accumulatively calculate the accuracy

  avg_loss = total_loss / len(data_loader)
  avg_accuracy = total_accuracy / len(data_loader)
  return avg_loss, avg_accuracy

def train(root, batch_size = 64, device = None, epochs = 3, save_path = None, train = True, test = True, visualize = True, predict = True):
    # device agnostic code
    if device is not None:
        # confirm device availability
        if torch.cuda.is_available():
            print("GPU Available")
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            print("GPU not available, using CPU")
            device = torch.device("cpu")

    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    val_dir = os.path.join(root, "val")

    train_dataloader = createDataLoader(train_dir, batch_size=batch_size)
    test_dataloader = createDataLoader(test_dir, batch_size=batch_size)
    val_dataloader = createDataLoader(val_dir, batch_size=batch_size)
    # create the model, optimizer and loss function
    model  = CrackDetectionModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    # train the model
    trainModel(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs= epochs, device= device)
    test_loss, test_accuracy = testModel(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model on custom data")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--predict", action="store_true", help="Predict on a single image")
    parser.add_argument("--visualize", action="store_true", help="Visualize the data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
    parser.add_argument("--device", type=str, default=None, help="Device to run the prediction on")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.train:
        if args.root is not None:
            train(args.root, args.batch_size, args.epochs, args.save_path)
        else:
            print("Root directory not provided")
            sys.exit(1)
    if args.test:
        if args.root is not None:
            testModel(args.root, args.batch_size)
        else:
            print("Root directory not provided")
            sys.exit(1)
    if args.visualize:
        if args.root is not None:
            visualizeImages(args.root, args.batch_size)
        else:
            print("Root directory not provided")
            sys.exit(1)
    if args.predict:
        if args.image_path is not None:
            predict_single_image(args.image_path, args.model_path, args.device)
        else:
            print("Image path not provided")
            sys.exit(1)