import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- CONFIG ---
DATA_ROOT = "./augmented_data"
BATCH_SIZE = 16 # higher -> more RAM usage
EPOCHS = 60
LEARNING_RATE = 0.001
NUM_WORKERS = 6 # higher -> more RAM usage, less GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True #faster for fixed image size
SAVE_DIR = "./models/checkpoints/"
# --------------

os.makedirs(SAVE_DIR, exist_ok=True)

#pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#checks for subfolders
full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)

classes = full_dataset.classes
num_classes = len(classes)

train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True)

#model hyperparameters
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out) 
        out = out.view(out.size(0), -1)
        return self.linear(out)

#training
if __name__ == '__main__':
    print(f"Found {num_classes} classes: {classes}")
    print(f"Total images: {len(full_dataset)}")

    model = ResNet18(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        
        #gemini created progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") #"batch" is just visual it means iterations/s
        
        # progress bar wraps train_loader, it holds the data it just also displays a bar
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update the progress bar text with the current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        torch.save(model.state_dict(), f"./models/checkpoints/checkpoint_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "./models/face_model_final.pth")
    print("\nFinished. Model saved as face_model_final.pth")