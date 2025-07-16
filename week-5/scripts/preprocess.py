import torchvision.transforms as transforms
import torchvision
import os

transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
dataset = torchvision.datasets.ImageFolder(root='data/raw/PokemonData',transform=transform)
i=1
for data in dataset:
    path = f'./data/processed/{data[1]}/image_{i}.jpg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(data[0], path)
    i+=1
