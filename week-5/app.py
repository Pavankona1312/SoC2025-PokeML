import gradio as gr
import torch
from torchvision import transforms
from pokemon_classifier import CNN

device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load("./trained_model-2.pth", map_location=torch.device('cpu')))
model.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
with open("pokemon.txt", "r") as f:
    pokemon_list = f.read().splitlines()

def predict(image):
    image = image.convert('RGB')
    input = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(input)
        p = torch.nn.functional.softmax(output, dim=1)[0]
        top5_p, top5_i = torch.topk(p, 5)
        results = {}
        for p, i in zip(top5_p, top5_i):
            results[pokemon_list[i]] = f"{p.item():.3f}"
    return results

interface = gr.Interface(fn=predict,inputs=gr.Image(type="pil"),outputs=gr.Label(num_top_classes=5),title="Pokemon Classifier",description="Upload a Pokemon image to get top-5 predictions!")
interface.launch(share=True)