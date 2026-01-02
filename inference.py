
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image
import cv2
import argparse
import os

class Convolutional_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1= nn.Conv2d(3,12,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(12,24,3)
        self.full1=nn.Linear(21384,120)
        self.full2=nn.Linear(120,84)
        self.full3=nn.Linear(84,3)

    def forward(self,x):
        x= self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x= torch.flatten(x,1)
        x= F.relu(self.full1(x))
        x= F.relu(self.full2(x))
        x= self.full3(x)
        return x                 
   


transform = tf.Compose([
    
    tf.Resize([140,116]),
    tf.ToTensor(),
    tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def webcam_inference(model, class_names, device='cpu'):
   
    model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    
    print("Webcam inference started! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = class_names[predicted.item()]
            confidence = torch.softmax(outputs, 1).max().item() * 100
        
        
        cv2.putText(frame, f"{pred_class}: {confidence:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Capture Live', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument('--model', required=True, help='Path to trained model .pth')
    
    parser.add_argument('--webcam', action='store_true', help='Run webcam inference')
    parser.add_argument('--classes', nargs='+', help='Class names list')
    
    args = parser.parse_args()
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Convolutional_Layer()  
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device)
    
    
    class_names = args.classes or ['Fake Oracle', 'Real Oracle', 'Roboccon Logo']  
    
    if args.webcam:
        webcam_inference(net, class_names, device)
    
    else:
        print("Use: python inference.py --model model.pth --webcam")
        print("  or: python inference.py --model model.pth --image test.jpg --classes QR0 QR1")

if __name__ == "__main__":
    main()
