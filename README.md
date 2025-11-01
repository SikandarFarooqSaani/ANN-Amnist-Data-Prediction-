# 🧠 MNIST Digit Classification using Artificial Neural Network (ANN)
**📘 AI-Generated README (Custom prompt used for clarity and learning ease)**  

---

## 📌 Overview  
This is the **first Deep Learning (DL)** or **toy project**, focused on building an **Artificial Neural Network (ANN)** to predict handwritten digits using the **MNIST dataset**.  
The goal is to classify grayscale 28×28 pixel images of digits (0–9) using a simple neural network built with **TensorFlow** and **Keras**.

---

## 🧰 Libraries Used  
- **TensorFlow / Keras** — for building and training the ANN  
- **NumPy** — for numerical operations  
- **Matplotlib** — for visualization  

---

## 📂 Dataset  
- **Dataset:** MNIST Handwritten Digit Dataset  
- **Training Samples:** 60,000  
- **Test Samples:** 10,000  
- **Image Size:** 28×28 pixels (grayscale)  
- Each pixel value ranges from **0 to 255** representing brightness.  

---

## 🧩 Step-by-Step Process  

### 1. Importing Required Libraries  
Imported:
- `tensorflow`
- `keras.models.Sequential`
- `keras.layers.Flatten`, `Dense`
- `matplotlib.pyplot as plt`

---

### 2. Data Loading and Inspection  
- Loaded the MNIST dataset using `keras.datasets.mnist.load_data()`.  
- Split into **x_train, y_train, x_test, y_test**.  
- Displayed the **2nd image** from training data using `plt.imshow()` → It was the **digit “4”** (📸 *<img width="416" height="413" alt="amnist1" src="https://github.com/user-attachments/assets/4f40d25b-45dd-496a-9e3e-a5777b95121c" />
*).  

---

### 3. Data Preprocessing  
- **Normalization:** Pixel values divided by 255 to scale between **0 and 1**, making computation faster and improving gradient descent performance.  
- **Shape:** Training data — (60000, 28, 28) → Flattened to 784 neurons per image for ANN input.  

---

### 4. Building the ANN Model  
- Created a **Sequential model** with the following architecture:  
  1. **Flatten Layer** → Converts 28×28 input into a 784-length vector.  
  2. **Dense Layer 1:** 128 neurons, activation = **ReLU**  
  3. **Dense Layer 2:** 32 neurons, activation = **ReLU**  
  4. **Output Layer:** 10 neurons, activation = **Softmax** (for multiclass classification)  

- Used `model.summary()` to visualize the architecture and parameters.

---

### 5. Compiling the Model  
- **Loss Function:** `sparse_categorical_crossentropy`  
- **Optimizer:** `Adam`  
- **Metrics:** `accuracy`  

---

### 6. Training the Model  
- Model trained using:  
  - **Epochs:** 15  
  - **Validation Split:** to observe validation accuracy/loss during training  
- Training progress stored in a variable `history` for analysis.

---

### 7. Model Evaluation  
- **Final Accuracy:** ✅ **0.97 (97%)** on test data.  
- Model performed excellently for a simple ANN.  

---

### 8. Predictions  
- Predictions are probabilities returned by `model.predict()`.  
- Used `argmax()` to get the class with highest probability.  
- Tested on an image from test set → correctly predicted **digit 2** (📸 *<img width="416" height="413" alt="amnist3" src="https://github.com/user-attachments/assets/a792fcf0-8bea-4479-84ea-bbf167b80997" />
*).  

---

### 9. Visualization  
- **Training History Plots:**  
  - Loss and validation loss decreased steadily and then stabilized (📸 *<img width="547" height="413" alt="amnist2" src="https://github.com/user-attachments/assets/1e567ced-1a02-4606-9279-638f83af334d" />
*).  
  - Accuracy improved and remained consistent after several epochs.  

---

## 🧠 Key Learnings  
- Normalizing image pixel values significantly improves training stability.  
- Simple ANN models can achieve high accuracy on structured datasets like MNIST.  
- Visualization of training history helps detect overfitting or underfitting.  
- Flattening is crucial for feeding image data into fully connected layers.

---

## 🏁 Final Results  
| Metric | Value |
|--------|--------|
| **Test Accuracy** | **97%** |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Hidden Layers** | 2 (128 & 32 neurons) |

---

📎 *Note: This README was AI-generated using a custom prompt for educational and clarity purposes.*
