import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from model import Model
from pathlib import Path
import sys

model = Model(5)
model_weights = torch.load(sys.argv[1])
model.load_state_dict(model_weights)
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

model.eval()

image = Image.open(Path(sys.argv[2]))
input_image_loader = DataLoader(dataset=image, batch_size=1, shuffle=False, num_workers=5)
input_image_loader = trans(image)
input_image_loader = input_image_loader.view(1, 3, 32,32)

labels_dict = {
    0: "daisy",
    1: "dandelion",
    2: "rose",
    3: "sunflower",
    4: "tulip"
}

pred_tensor = model(input_image_loader)
print(pred_tensor)
prediction = pred_tensor.data.numpy().argmax()
print("Predicted class:", labels_dict[prediction])