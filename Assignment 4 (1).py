#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[8]:


#Q1
from sklearn. datasets import load_breast_cancer
from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

# Shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# this data is imbalanced as there are 357 benign cases yet only 212 malignant
# class balance is important to make sure that every class is bein analyzed equally


# In[9]:


#Q2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, dt.predict(X_test))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# entorpy measures how good a partition is
# the test accuracy is lower showing that there was overfitting


# In[10]:


#Q3
dt_constrained = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=42) # constraint

dt_constrained.fit(X_train, y_train)

train_acc_c = accuracy_score(y_train, dt_constrained.predict(X_train))
test_acc_c = accuracy_score(y_test, dt_constrained.predict(X_test))

print("Train Accuracy (constrained):", train_acc_c)
print("Test Accuracy (constrained):", test_acc_c)

# Feature importance
import pandas as pd

feature_importance = pd.Series(
    dt_constrained.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

print("Top 5 Features:\n", feature_importance.head())

# constraining data like giving it a max depth reduces over fitting this is because if you had a unlimmited max depth then the splits would be a lot more specific
# Feature importance matters because these are the  features that have the most influence on predictions.


# In[11]:


#Q4
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
neural_network_model = Sequential()
neural_network_model.add(InputLayer(input_shape=(30,)))
neural_network_model.add(Dense(8))
neural_network_model.add(Dense(1, activation='sigmoid'))

# Compile
neural_network_model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
neural_network_model.fit(
    X_train_scaled,
    y_train,
    epochs=20
)

# Evaluate
train_loss, train_acc = neural_network_model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc = neural_network_model.evaluate(X_test_scaled, y_test, verbose=0)

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
# Feature scaling is necessary because neural networks learn faster and more effectively


# An epoch represents one complete pass through the entire training dataset.


# In[12]:


#Q5
from sklearn.metrics import confusion_matrix

# Decision Tree
cm_dt = confusion_matrix(y_test, dt_constrained.predict(X_test))
print("Decision Tree Confusion Matrix:\n", cm_dt)

# Neural Network
y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype(int)
cm_nn = confusion_matrix(y_test, y_pred_nn)
print("Neural Network Confusion Matrix:\n", cm_nn)

# For this task I would prefer the Decision Tree model. It gives good accuracy and is easy to interpret.

# Decision Tree:
# Advantage: Highly interpretable; you can see which features and thresholds determine the predictions.
# Limitation: Can overfit if unconstrained; may not generalize well to unseen data without proper constraints.

# Neural Network:
# Advantage: Can capture complex patterns and interactions between features,potentially achieving high accuracy.
# Limitation: Acts as a "black box"; it is hard to interpret how it makes decisions.


# In[14]:


#Q6
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import fashion_mnist

# Step 1: Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Step 2: Normalize pixel values to range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 3: Reshape images to include channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 4: Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
])

# Step 5: Add classifier layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

# Step 6: Compile model 
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Step 7: Train model
history = model.fit(X_train, y_train,validation_split=0.1,epochs=15,batch_size=64)

# Step 8: Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("CNN Test Accuracy:", test_acc)

# CNNs are generally preferred over fully connected networks for image data because:
# 1. They take advantage of the spatial structure of images (neighboring pixels are related).
# 2. They use fewer parameters through local connections and weight sharing, which reduces
#    memory usage and overfitting.

# The convolution layer is learning small patterns or features in the images, such as edges,
# corners, or textures, which are combined in deeper layers to recognize more complex shapes.


# In[15]:


#Q7
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model_cnn.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("CNN Confusion Matrix")
plt.show()

# Misclassified examples
misclassified = np.where(y_pred_classes != y_test)[0]

for i in range(3):
    idx = misclassified[i]
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.show()
# One pattern in the misclassifications: the CNN often confuses visually similar classes,
# such as "Shirt" vs "T-Shirt" or "Sandal" vs "Sneaker", because their shapes and textures overlap.

# One realistic method to improve CNN performance: use data augmentation (e.g., rotations,
# flips, or shifts) to create more varied training images, which helps the model generalize better.


# In[ ]:




