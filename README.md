# Siamese Network Face Recognition

This project implements a Face Recognition system using a **Siamese Neural Network** built with **TensorFlow/Keras**.  
It performs **one-shot facial recognition** by computing the distance between an **anchor image** and a **verification image**.

---

##  Project Overview

Unlike traditional classification models that require thousands of images per identity, a **Siamese Network** learns a *similarity function*.  
It takes **two images** as input and outputs a similarity score:

- **1 (True)** → The two faces belong to the **same person**
- **0 (False)** → The two faces belong to **different people**

A threshold (commonly **0.5**) is used to decide whether faces match.

---

##  Tech Stack

- **Language:** Python 3.x  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV (cv2)  
- **Data Manipulation & Visualization:** NumPy, Matplotlib  

---

##  Project Structure

├── data/
│   ├── anchor/      # Images captured from webcam (You)
│   ├── positive/    # Images captured from webcam (You)
│   └── negative/    # Images from external dataset (e.g., LFW)
├── application_data/
│   └── verification_images/
├── Face_Recognition.ipynb  # Main Jupyter Notebook
└── README.md


##  Data Setup

The project requires **three categories of images**:

- **Anchor:**  
  Image of the target person captured using the webcam.

- **Positive:**  
  Another image of the **same person** (also captured using the webcam).

- **Negative:**  
  Images of **different people**, downloaded from the **Labelled Faces in the Wild (LFW)** dataset.

###  Collecting Your Own Data

To collect training data, run the **data collection cells** inside the notebook.

Use the following keyboard shortcuts:

- Press **`a`** → Save an **Anchor** image  
- Press **`p`** → Save a **Positive** image  

##  Model Architecture

The model uses a standard **Siamese Network** architecture consisting of:

- **Twin Networks:**  
  Two identical Convolutional Neural Networks (CNNs) that **share the same weights**.

- **Embeddings:**  
  Each network outputs a **4096-dimensional feature vector** (embedding).

- **L1 Distance Layer:**  
  Calculates the absolute difference between the two embeddings:  
  `|Embedding_1 - Embedding_2|`.

- **Prediction Layer:**  
  A dense layer with **sigmoid activation** outputs a similarity score between **0 and 1**.
  
<img width="683" height="234" alt="image" src="https://github.com/user-attachments/assets/eb247dbe-81c2-433c-b72b-5d678a3b9157" />

##  Training

Open `Face_Recognition.ipynb` and run the training section.

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Precision, Recall  

The model saves checkpoints during training to allow you to **resume** or **reuse** the trained model later.

---

## Real-Time Usage

The project includes a real-time verification script using your webcam:

1. The system captures a frame from the webcam.  
2. It compares this frame against a folder of **verification images**.  
3. If the verification threshold is met (e.g., > 0.5 detection confidence **and** > 0.5 verification accuracy), **access is granted**.

---

##  References

- Based on the paper:  
  *Siamese Neural Networks for One-shot Image Recognition* by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Dataset: [Labelled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)




