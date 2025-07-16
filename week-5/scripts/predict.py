import torch 
from torchvision import transforms
from pokemon_classifier import CNN
import argparse

model = CNN()
model.load_state_dict(torch.load("../models/trained_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the image to classify.')
args = parser.parse_args()

image_path = args.image_path
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

with open("../pokemon.txt", "r") as f:
    pokemon_list = f.read().splitlines()

from PIL import Image
image = Image.open(image_path).convert('RGB')
input = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    model.eval()
    output = model(input)
    p = torch.nn.functional.softmax(output, dim=1)[0]
    top5_p, top5_i = torch.topk(p, 5)
    for p, i in zip(top5_p, top5_i):
        print(f"{pokemon_list[i]}: {p.item():.3f}")






