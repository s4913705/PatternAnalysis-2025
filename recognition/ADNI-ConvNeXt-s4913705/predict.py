import torch
from torchvision import transforms
from PIL import Image
from modules import ConvNeXtADClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvNeXtADClassifier()
model.load_state_dict(torch.load("/content/drive/MyDrive/best_convnext_adni_optimized.pth"))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, 1).item()
    print(f"Prediction: {['AD', 'NC'][pred]}")

# Example usage:
# predict_image("/content/drive/MyDrive/ADNI/AD_NC/test/AD/sample1.jpg")
