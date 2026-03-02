import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# --- CONFIG ---
MODEL_PATH = "./models/two_class_model.pth"
CLASSES = ["one_class", "two_class"] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FPS = 16 # doesn't need to match original fps
INTERVAL = 1.0 / TARGET_FPS
# --------------

# --- MODEL ARCHITECTURE ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# --------------------------

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

# --- INITIALIZE AND LOAD ---
model = ResNet18(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ---------------------------

cap = cv2.VideoCapture(0)
print(f"Inference started on {DEVICE}. Press 'q' to quit.")

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        #resize frame, make standard
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

        #prediction calculation
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
            
            label = CLASSES[pred.item()]
            confidence = conf.item() * 100

        # if >80% confidence green, otherwise red
        color = (0, 255, 0) if confidence > 80 else (0, 0, 255)
        
        #display text
        cv2.putText(frame, f"Pred: {label}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Conf: {confidence:.1f}%", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        #debug
        #print(f"Seeing: {label} ({confidence:.1f}%)")

        # show frame
        cv2.imshow("Classifier", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

        elapsed = time.time() - start_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

finally:
    cap.release()
    cv2.destroyAllWindows()