import os # for kaggle archive
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# training our logistic regression 
clf = LogisticRegression(max_iter=500, solver='lbfgs')
print("Training model...")
clf.fit(X_train, y_train)

# final results 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy (Cats vs Dogs): {accuracy * 100:.2f}%")



