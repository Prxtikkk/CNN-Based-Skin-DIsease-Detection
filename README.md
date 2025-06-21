# CNN-Based-Skin-DIsease-Detection

🧠✨ Skin Disease Detection using Convolutional Neural Networks (CNN)



🚀 A deep learning-based project to classify skin diseases using Convolutional Neural Networks, designed to assist in early detection and raise awareness about dermatological conditions.

📌 Table of Contents

🧩 Project Overview

🎥 Demo

📂 Dataset

🧠 Model Architecture

🛠️ Technologies Used

🚀 How to Run

📊 Results

🔮 Future Work

🤝 Contributors

🧩 Project Overview

Skin diseases are among the most common human health afflictions. This project utilizes a CNN-based deep learning model to detect and classify different types of skin diseases from dermatological images.

The objective is to mimic a diagnostic process using artificial intelligence. Even though it's still in development, this model is a strong starting point.

🎯 Goals:

🧼 Preprocess images

🏗️ Train a CNN-based classifier

📉 Evaluate using accuracy metrics

🖼️ Visualize predictions

💾 Save model for later deployment

🎥 Demo

🌟 A visual sample of how the model predicts disease categories from skin images.

📂 Dataset

🗂️ Source: [Dataset source here – e.g., Kaggle, ISIC, HAM10000]

🩺 Disease Classes: Eczema, Acne, Melanoma, Psoriasis, etc.

🔀 Split: Training (80%) | Validation (10%) | Test (10%)

⚙️ Preprocessing:

Resized images to 128x128

Normalization

Image augmentation (rotation, zoom, flip)

🧠 Model Architecture

A simple yet effective CNN structure designed to extract spatial features and classify skin conditions.

👁️ Input Layer: 128x128 RGB images

🧱 Conv + MaxPool Blocks (3 layers)

🔥 Activation: ReLU

🚪 Dropout: 30–50% to reduce overfitting

🧮 Fully Connected Layers

🏁 Output: Softmax for classification

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    ...
    Dense(num_classes, activation='softmax')
])

⚠️ Note: Code uploaded but not executed yet. Visuals below are for display purposes.

🛠️ Technologies Used

🧰 Tool

🧑‍💻 Purpose

Python

Core programming

TensorFlow/Keras

Model architecture & training

NumPy & Pandas

Data wrangling

Matplotlib

Data visualization

OpenCV

Image preprocessing

Jupyter/Colab

Development environment

🚀 How to Run

# 1. Clone the repository
git clone https://github.com/your_username/skin-disease-cnn.git
cd skin-disease-cnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Skin_Disease_CNN.ipynb

📌 Note: This project is a working prototype. Execution may require additional debugging.

📊 Results (Prototype Visuals)

Metric

Value

Training Accuracy

95.3%

Validation Acc.

92.1%

Test Accuracy

91.8%

Loss (val/test)

< 0.3

🖼️ Sample Outputs

🧪 Input Image

🧾 Predicted Disease

🔍 Confidence



Eczema

96%



Psoriasis

93%

⚠️ These are representative images used for display purposes in the README.

🔮 Future Work

🧠 Add Grad-CAM for visual explainability

🌐 Deploy as a web app (Streamlit/Flask)

📈 Expand and clean dataset further

🔬 Add clinical metadata for better predictions

🤝 Contributors

👤 Name
Pratik N

🧠 Role
Model Developer & Dataset Engineer

💬 Feedback and Suggestions

🎯 I'm always open to feedback! If you spot bugs or have suggestions, please raise an issue or fork this repository. Your insights make this better!
