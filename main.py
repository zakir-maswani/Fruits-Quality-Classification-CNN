
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Defien the architecture of CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16*16*128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
# Create model
model = CNN()
model.load_state_dict(
    torch.load(
        "fruit_quality_and_adulteration_classifier.pth"
    )
)

model.eval()

class_names = ['Fresh', 'Rotten']

# Transformation pipeline

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

app = FastAPI(title="Fruits Disease Classifier API")

app.mount("/static", StaticFiles(directory= "static"), name="static")
templates = Jinja2Templates(directory= "templates")

# Serve home
@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse(request= request, name="index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidance, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class_name = class_names[
        predicted_class.item()
    ]

    confidence_percentage = (
        confidance.item() * 100
    )
    

    return {
        "prediction": predicted_class_name,
        "confidence": round(
            confidence_percentage,
            2
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)