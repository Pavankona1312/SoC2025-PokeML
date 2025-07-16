# WEEK 5: Pokémon IMAGE CLASSIFIER

## Introduction
This project implements a deep learning-based image classifier that identifies Pokémon from images using a custom CNN architecture we created. The classifier is trained on a 150-class Pokémon image dataset and offers Gradio-powered interface for predictions.

---

##  Dataset

- **Source**: [Kaggle - Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- **Structure**: 
  - Raw images stored in `data/raw/PokemonData/`
- **Preprocessing**:
  - Only resized the images
        
To preprocess the data, run:
```bash
python scripts/preprocess.py
```

---

##  Model Architecture

The CNN is implemented in `pokemon_classifier.py` and consists of:

- **3 Convolutional Blocks**:
  - Conv2D → BatchNorm → LeakyReLU → MaxPool2d
- **Fully Connected Layers**:
  - Flatten → Linear(9216 → 1024 → 512 → 256 → 150)
  - BatchNorm, LeakyReLU, and Dropout in each layer

Final layer outputs logits for 150 Pokémon classes.

---

##  Training Details

- **Loss Function**: `CrossEntropyLoss`  
- **Optimizer**: `Adam`  
- **No.of.epochs**: 50
- **Batch size**: 32
- **Learning rate**: 0.01
  
 | Metric              | Value (Example)   |
|---------------------|------------------|
| Training Loss       | ~0.857          |
| Validation Accuracy | ~64.66%            |
| Test Accuracy       | ~62.68%            |

---

## Graphs
- Training accuracy vs. Epochs
  
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/98f63315-90d5-470d-9242-45802cc1a54b" />

- Training loss vs. Epochs
  
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/26e3d15a-cf3d-4307-bf1f-4ba64d3992c8" />


### To Train:
```bash
python scripts/train.py 
```
---

## Challenges faced and how we dealt it with:

Early epochs showed significant overfitting; resolved partially via:

- Data augmentation: RandomHorizontalFlip
- Regularization: Dropout(p=0.5) in fully connected layers
- Batch Normalization in both convolutional and linear layers
  
Initial test accuracy was below 50%; tuning learning rate, increasing epochs, and using LeakyReLU helped improve it.

---

## Gradio Frontend

An interactive Gradio app allows you to upload a Pokémon image and see predictions in real-time.

### How to launch the App?
```bash
python app.py
```

### Features:
- Upload an image
- Get top-5 predicted Pokémon classes
- Clean and simple interface

**Example Screenshot**:

<img width="1280" height="703" alt="image" src="https://github.com/user-attachments/assets/2c4f67d3-3525-4bbf-8f02-53928831231c" />

---

## How to Use This Project

1. **Clone the Repository**:
```bash
git clone <your_repo_url>
cd week_5
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Dataset**:
Place the Pokémon dataset in:
```
data/raw/PokemonData/
```

4. **Preprocess the Data**:
```bash
python scripts/preprocess.py
```

5. **Train the Model**:
```bash
python scripts/train.py
```

6. **Run Inference**:
```bash
python scripts/predict.py --image_path path/to/image.jpg
```

7. **Launch Gradio App**:
```bash
python app.py
```

---

## Dependencies

Install all with:
```bash
pip install -r requirements.txt
```

---


> Project by Pavan and Dhanvin

