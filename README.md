# 🧠 Brain Tumor Detection with Gamma-Based Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/Deep_Learning-Keras-orange.svg)
![UI](https://img.shields.io/badge/Interface-Tkinter-lightgrey.svg)

> A deep learning-based application that detects brain tumors in MRI images using CNN and gamma-based edge analysis. Includes a GUI built with Tkinter for model training, testing, and visualization.

---

## 📊 Overview

This project focuses on classifying brain MRI images into **tumor** or **no tumor** categories using a Convolutional Neural Network (CNN). It also applies **gamma edge analysis** to highlight the tumor region. Key components include:

- 🧠 CNN model for binary image classification
- 🎯 Gamma-based segmentation for edge highlighting
- 📈 Accuracy visualization via training graphs
- 🖥️ Interactive GUI for dataset upload, prediction, and graphing

---

## 📁 Project Structure

```bash
brain-tumor-detection/
├── Model/
│   ├── model.json                # Saved model architecture
│   ├── model_weights.h5          # Trained CNN weights
│   ├── history.pckl              # Accuracy history for plotting
│   ├── myimg_data.txt.npy        # Numpy dataset file
│   ├── myimg_label.txt.npy       # Numpy label file
├── testImages/                   # Sample test MRI images
├── test1.png                     # Resized original image (used in prediction)
├── myimg.png                     # Model's predicted tumor mask
├── 1.png, 2.png, 97.png          # Sample tumor image examples
├── New Text Document.txt         # Main Python script (GUI + Model)
└── README.md                     # Project documentation
```

---

##🚀 Getting Started
1.Clone the repository
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```
2.Install the required packages
```bash
pip install -r requirements.txt
```
3.Run the application
```bash
python "New Text Document.txt"
```
The GUI will launch, allowing you to upload images, build the model, and test predictions interactively.

---

## 📸 Visual Previews

<table>
  <tr>
    <td><img src="1.png" alt="Sample Tumor Image 1" width="250"/></td>
    <td><img src="2.png" alt="Sample Tumor Image 2" width="250"/></td>
    <td><img src="97.png" alt="Sample Tumor Image 3" width="250"/></td>
  </tr>
  <tr>
    <td align="center">Tumor Image 1</td>
    <td align="center">Tumor Image 2</td>
    <td align="center">Tumor Image 3</td>
  </tr>
</table>

> Images processed by the model, showing regions affected by tumors through gamma-based edge analysis.

---

## 🧪 Model Evaluation

- ✅ Dice Coefficient Accuracy Measurement  
- 🔍 Visual Analysis of Predicted Masks  
- 📈 Graph of Training Accuracy per Epoch  
- 🎯 GUI outputs accuracy and prediction status live

---

## 🧹 Data Flow & Preprocessing

- 📂 Upload MRI dataset via Tkinter file dialog  
- 🌀 Resize & normalize input images  
- 🧠 CNN Model built or loaded from saved files  
- 🔄 Predict tumor presence on test images  
- 🎨 Gamma Edge Analysis for visualization

---

## 🧠 Architecture Highlights

- Multi-layer CNN with:
  - `Conv2D ➡ MaxPooling ➡ Flatten ➡ Dense`
- Optional **U-Net** variant for segmentation
- Activation functions: **ReLU** + **Softmax/Sigmoid**
- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## 🙌 Acknowledgments

Thanks to the **open-source Python** and **Keras** communities, and all contributors in the **medical imaging research** space.

---

## 📫 Contact

For questions, suggestions, or collaborations:  
📧 **akhilsai96@gmail.com**


