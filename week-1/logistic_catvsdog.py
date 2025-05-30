import os # for kaggle archive
from PIL import Image
import numpy as np

# accessing the archive
base_folder = "archive"
train_cat_folder = os.path.join(base_folder, "train", "cats")
train_dog_folder = os.path.join(base_folder, "train", "dogs")
test_cat_folder = os.path.join(base_folder, "test", "cats")
test_dog_folder = os.path.join(base_folder, "test", "dogs")

# loading the images
def load_images_from_folder(folder, label, size=(64, 64)):
    images, labels = [], []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            img = Image.open(fpath).resize(size).convert('RGB')
            img_np = np.array(img) / 255.0
            images.append(img_np.flatten())
            labels.append(label)
        except Exception as e:
            print(f"Skipped {fname}: {e}")
    return images, labels

# training data images 
print("Loading training images...")
X_cat_train, y_cat_train = load_images_from_folder(train_cat_folder, 0)
X_dog_train, y_dog_train = load_images_from_folder(train_dog_folder, 1)

X_train = np.array(X_cat_train + X_dog_train)
y_train = np.array(y_cat_train + y_dog_train)

# test data images 
print("Loading test images...")
X_cat_test, y_cat_test = load_images_from_folder(test_cat_folder, 0)
X_dog_test, y_dog_test = load_images_from_folder(test_dog_folder, 1)

X_test = np.array(X_cat_test + X_dog_test)
y_test = np.array(y_cat_test + y_dog_test)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# training logistic regression from scratch
def train_logistic(X, y, lr=0.1, epochs=100):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for epoch in range(epochs):
        z = np.dot(X, w) + b
        a = sigmoid(z)
        dz = a - y
        dw = np.dot(X.T, dz) / m
        db = np.sum(dz) / m

        w -= lr * dw
        b -= lr * db

        if epoch % 10 == 0:
            loss = -np.mean(y * np.log(a + 1e-8) + (1 - y) * np.log(1 - a + 1e-8))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b

# predict function
def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= 0.5).astype(int)

# training our logistic regression 
print("Training model...")
w, b = train_logistic(X_train, y_train, lr=0.1, epochs=100)

# final results 
y_pred = predict(X_test, w, b)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy (Cats vs Dogs): {accuracy * 100:.2f}%")
