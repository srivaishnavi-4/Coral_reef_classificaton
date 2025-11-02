# 🪸 Coral Reef Health Classification – Healthy vs Bleached

## 📖 Overview

This project uses image classification with convolutional neural networks (CNNs) to distinguish between **healthy** and **bleached** coral images. The goal is to contribute to marine ecosystem sustainability by creating a tool that helps monitor coral health automatically.

---

## 🌍 Problem Statement

Coral reefs are vital to marine biodiversity but are increasingly threatened by climate change and environmental stressors that cause bleaching. Manual monitoring is costly and time-intensive. By building an automated classification model, we can assist conservation efforts, enabling faster screening of coral imagery for bleaching.

---

## 🎯 Objectives

* Build and train a CNN (or use transfer learning) to classify coral images into two categories: *Healthy Coral* and *Bleached Coral*.
* Evaluate model performance with metrics such as accuracy, confusion matrix, precision, recall and F1-score.
* Visualize model predictions and provide insight into how the model distinguishes between healthy and bleached conditions.
* Document the project for reproducibility and possible deployment (e.g., a simple web or mobile interface).

---

## 📂 Dataset

**Dataset Name:** Coral Reefs Images – Healthy & Bleached
**Source:** [Kaggle dataset](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images)
**Classes:**

1. Healthy
2. Bleached

**Suggested Folder Structure:**

```
/data/
   /train/
       /healthy/
       /bleached/
   /validation/
       /healthy/
       /bleached/
   /test/
       /healthy/
       /bleached/
```

---

## ⚙️ Tools & Technologies

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (or PyTorch if preferred)
* **Image Processing:** OpenCV, PIL
* **Visualization:** Matplotlib, Seaborn
* **Optional UI/Deployment:** Streamlit, Flask
* **Environment:** Jupyter Notebook or Google Colab

---

## 🧠 Model Architecture

For example, using transfer learning with a pre-trained CNN such as EfficientNetB0:

* Base model: EfficientNetB0 (pre-trained on ImageNet)
* Freeze base layers initially
* Add a Global Average Pooling layer
* Add Dense (128 units, ReLU) + Dropout (e.g., 0.3)
* Output layer: Dense(2) with Softmax (since two classes)
* Loss function: Categorical Crossentropy (or Binary Crossentropy if using two-class single output)
* Optimizer: Adam
* Metrics: Accuracy, Precision, Recall, F1-score

---

## 🧾 Project Workflow

1. **Data Preparation**

   * Load and inspect the dataset
   * Split into train/validation/test sets
   * Resize images (e.g., 224×224)
   * Normalize pixel values (e.g., divide by 255)
   * Apply data augmentation (rotation, flips, brightness shifts) to improve generalization

2. **Model Building & Training**

   * Set up the base model with transfer learning
   * Compile the model with chosen loss/optimizer/metrics
   * Train over a number of epochs (e.g., 15-30)
   * Use callbacks: EarlyStopping, ModelCheckpoint

3. **Evaluation**

   * Evaluate the trained model on the validation and test sets
   * Generate a confusion matrix, classification report (precision, recall, F1)
   * Plot accuracy & loss curves
   * (Optional) Use visualization techniques like Grad-CAM to highlight what parts of the image influenced the model’s decision

4. **Deployment or Demo (Optional)**

   * Create a simple web interface where a user can upload a coral image and get a prediction (Healthy or Bleached)
   * Use Streamlit or Flask for quick UI

---

## 📊 Expected Results

| Metric              | Approximate Value*    |
| ------------------- | --------------------- |
| Training Accuracy   | ~ 90-95%              |
| Validation Accuracy | ~ 85-90%              |
| Test Accuracy       | similar to validation |

*Actual values will depend on dataset size, augmentation, model choice, and hyper-parameters.

Visualization outputs:

* Accuracy & Loss vs Epoch graphs
* Confusion Matrix
* Example images with predicted class and probability
* (If doing Grad-CAM) heatmaps showing attention regions

---

## 💡 Key Learnings

* Transfer learning significantly boosts performance especially when dataset size is moderate.
* Data augmentation is critical, especially with underwater images which vary in lighting, turbidity, and color.
* Choosing the right model and fine-tuning can help discriminate subtle differences (healthy vs bleached coral).
* Visual explanations (e.g., Grad-CAM) help trust and interpret the model’s decisions — important for real-world ecological use.

---

## 🌊 Sustainability Impact

* Supports automated monitoring of coral reefs — increases scalability and lowers manual effort.
* Early detection of bleaching events can prompt quicker responses by conservation teams.
* Contributes to the broader goal of protecting marine ecosystems and biodiversity.

---

## 🚀 Future Enhancements

* Extend to **three-class classification** (Healthy, Bleached, Dead) if data allows.
* Incorporate **segmentation models** (e.g., U-Net) to not just classify, but **localize** bleached regions within an image.
* Use **drone or satellite imagery** for large-scale reef monitoring.
* Develop a **mobile app** to allow divers or citizen scientists to contribute images and get on-site feedback.
* Explore **domain adaptation** if applying the model to different geographic regions (corals look different in various oceans).

---

## 🧩 Folder Structure

```
Coral-Reef-Health-Classification/
│
├── data/                   # raw images and train/val/test splits  
├── notebooks/              # Jupyter/Colab notebooks for development  
├── models/                 # saved model weights (e.g., model.h5)  
├── results/                # plots: accuracy/loss, confusion matrix  
├── app.py                  # optional Streamlit/Flask app for deployment  
├── README.md               # project documentation  
└── requirements.txt        # Python dependencies  
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Coral-Reef-Health-Classification.git  
cd Coral-Reef-Health-Classification  
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt  
```

### 3. Run Training

```bash
python train.py  
```

### 4. Launch Demo App (Optional)

```bash
streamlit run app.py  
```

or

```bash
python app.py  
```

---

## 📄 Citation

If you use this project, dataset, or share results in a publication, please cite the dataset and any model-architectures appropriately.

