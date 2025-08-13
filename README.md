# vision-ai-image-recognition
End-to-end deep learning image recognition project using Python, TensorFlow, and Keras. Features data preprocessing, custom CNN, transfer learning with MobileNetV2, data augmentation, evaluation metrics, and visualizations. Built in the Vision AI bootcamp, ready for deployment.
---

## 🚀 Project Overview
The goal of this project is to classify images into predefined categories with high accuracy using deep learning.  
We leverage **MobileNetV2** for feature extraction and fine-tuning to achieve efficient and accurate predictions even with limited training resources.

---

## 🛠 Tech Stack
- **Language:** Python 3.x  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Model:** MobileNetV2 (Transfer Learning)  

---

## 📂 Dataset
- **Source:** *(CIFAR-10)*  
- **Format:** Images organized in class-labeled folders.  
- **Preprocessing:**  
  - Resized to **224x224** pixels  
  - Normalized pixel values  
  - Applied data augmentation (rotation, flipping, zoom, etc.)  

---

Final Model Performance Table:

Model               Accuracy  Precision Recall    F1-score  
Custom CNN          66.63     0.67      0.67      0.66      
Augmented CNN       65.86     0.66      0.66      0.65      
MobileNetV2 TL      79.70     0.81      0.80      0.80 
Usage
To use this code, follow these steps:

1. Clone the repository: git clone https://github.com/gittushar18/vision-ai-image-recognition.git
2. Navigate to the project directory: cd your-repo-name
3. Install the required dependencies (see below)
4. Run the preprocessing script: python preprocessing.py
5. Train the model: python train.py
6. Evaluate the model: python evaluate.py
7. Make predictions: python predict.py

Dependencies
This project requires the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-lear
