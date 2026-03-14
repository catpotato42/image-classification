import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from screeninfo import get_monitors

# --- CONFIG ---
MODEL_PATH = "./models/checkpoints/checkpoint_epoch_7.pth"
#These have to be alphabetical as that's the way the training loader goes through folders. 
CLASSES = ['crine_class', 'five_class', 'one_class', 'thinking_class', 'timeout_class', 'twin_class', 'two_class', 'working_class'] 
CLASS_PATHS = ['./output/crine_baby.jpg', './output/five_pibble.jpg', './output/one_nerd.png', './output/thinking_monkey.png', './output/timeout_shaq.jpg', './output/twin_affleck.jpg', './output/two_hamster.jpg', './output/working_incredibles.jpg']
final_images = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FPS = 16 #doesn't need to match original fps
INTERVAL = 1.0 / TARGET_FPS
# --------------

monitor = get_monitors()[0]
SCREEN_HEIGHT = monitor.height
SCREEN_WIDTH = monitor.width

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

def standardize_img(img):
    h, w = img.shape[:2]

    #force exact screen height
    scale = SCREEN_HEIGHT / h
    new_height = SCREEN_HEIGHT
    new_width = int(w * scale)

    #ensure width is <= half screen
    max_width = SCREEN_WIDTH // 2 #if odd make even
    if new_width > max_width:
        new_width = max_width

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

def standardize_img_output(img):
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

def standardize_webcam_img(img):
    h, w = img.shape[:2]
    #screen height over two
    scale = SCREEN_HEIGHT / (2*h)
    new_height = int(h * scale)
    new_width = SCREEN_WIDTH // 2
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    padding = int((SCREEN_HEIGHT - new_height) / 2) # i don't even think this works with odd screen height
    pad_top = ((padding, 0), (0, 0), (0, 0))
    pad_bottom = ((0, padding), (0, 0), (0, 0))
    img = np.pad(img, 
                pad_width=pad_top,
                mode='constant', 
                constant_values=0)
    img = np.pad(img, 
                pad_width=pad_bottom,
                mode='constant', 
                constant_values=0)
    return img

current_output = np.zeros((224, 224, 3), dtype=np.uint8)

def get_output () :
    return current_output

# --- INITIALIZE AND LOAD ---
model = ResNet18(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ---------------------------

cap = cv2.VideoCapture(0)
print(f"Inference started on {DEVICE}. Press 'q' to quit.")

for name in CLASS_PATHS:
    img = cv2.imread(name)
    
    std_img = standardize_img(img)
    
    final_images.append(std_img)

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

        #standardize frame, concatenate output
        frame = standardize_webcam_img(frame)
        i = 0
        while (i < len(final_images)) :
            if label == CLASSES[i] :
                frame = np.hstack((frame, final_images[i]))
                current_output = final_images[i]
                break
            i += 1

        cv2.imshow("Output to Image", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

        elapsed = time.time() - start_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

finally:
    cap.release()
    cv2.destroyAllWindows()