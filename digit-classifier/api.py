from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import os
from model import SimpleCNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

try:
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Warning: model.pth not found. Please train the model first.")

class ImageData(BaseModel):
    image: str

@app.post("/predict")
async def predict(data: ImageData):
    base64_str = data.image.split(",")[1] if "," in data.image else data.image
    img_bytes = base64.b64decode(base64_str)
    
    img = Image.open(BytesIO(img_bytes))
    
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
        
    img = img.convert('L')
    img = ImageOps.invert(img)
    
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    max_dim = max(img.size)
    pad = int(max_dim * 0.2)
    new_size = max_dim + 2 * pad
    square_img = Image.new('L', (new_size, new_size), 0)
    
    offset = ((new_size - img.size[0]) // 2, (new_size - img.size[1]) // 2)
    square_img.paste(img, offset)
    
    img = square_img.resize((28, 28), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        confidences = F.softmax(outputs, dim=1)[0]
        
    conf_list = confidences.cpu().numpy().tolist()
    predicted_class = int(torch.argmax(confidences).item())
    
    return {
        "prediction": predicted_class,
        "confidences": conf_list
    }

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
