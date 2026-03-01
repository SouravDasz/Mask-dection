import torch
import torch.nn as nn
import cv2
from torchvision import transforms


class Mycnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256),  
            nn.ReLU(),
            nn.Linear(256, 1) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model_path = r"E:\CV\Mask-dection\mask pehen.pth"
model = torch.load(model_path, map_location='cpu', weights_only=False)
model.eval()


camera = cv2.VideoCapture(0)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    ret, frame = camera.read()
    if not ret:
        break

    display_frame = frame.copy()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0) 

    with torch.no_grad():
        output = model(img)
        output = torch.sigmoid(output).item() 

    cls_name = "no Mask" if output >= 0.5 else "Mask"

    cv2.putText(display_frame, cls_name,
                (50, 50),
                font,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("web cam", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()