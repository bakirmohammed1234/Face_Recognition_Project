Siamese Network Face Recognition
This project implements a Face Recognition system using a Siamese Neural Network. It is built with TensorFlow/Keras and performs one-shot image recognition by calculating the distance between an anchor image and a verification image.

📌 Project Overview
Unlike traditional classification models that require thousands of images per class, this Siamese Network learns a "similarity function." It takes two images as input and outputs a similarity score:

1 (True): The faces belong to the same person.

0 (False): The faces belong to different people.

The model assumes a threshold (typically 0.5) to decide if the faces match.

🛠 Tech Stack
Language: Python 3.x

Deep Learning: TensorFlow, Keras

Computer Vision: OpenCV (cv2)

Data Manipulation: NumPy, Matplotlib

📂 Project Structure
Bash

├── data/
│   ├── anchor/      # Images captured from webcam (You)
│   ├── positive/    # Images captured from webcam (You)
│   └── negative/    # Images from external dataset (e.g., LFW)
├── application_data/
│   └── verification_images/
├── Face_Recognition.ipynb  # Main Jupyter Notebook
└── README.md
🚀 Getting Started
1. Prerequisites
Install the required dependencies:

Bash

pip install tensorflow opencv-python matplotlib numpy
(Note: If you have a compatible NVIDIA GPU, ensure tensorflow-gpu is configured for faster training.)

2. Data Setup
The project requires three types of data:

Anchor: Input image (webcam).

Positive: Matches the anchor (webcam).

Negative: Different people (downloaded from the Labelled Faces in the Wild (LFW) dataset).

To collect data: Run the data collection cells in the notebook.

Press a to save an Anchor image.

Press p to save a Positive image.

3. Model Architecture
The model uses a standard Siamese architecture:

Twin Networks: Two identical Convolutional Neural Networks (CNNs) share the same weights.

Embeddings: Both networks output a 4096-dimensional feature vector (embedding).

L1 Distance Layer: Calculates the absolute difference between the two embeddings: |Embedding_1 - Embedding_2|.

Prediction: A dense layer with a sigmoid activation outputs a score between 0 and 1.

4. Training
Open Face_Recognition.ipynb and run the training section.

Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Precision, Recall

The model saves checkpoints during training to allow you to resume or use the model later.

🎥 Real-Time Usage
The project includes a real-time verification script using your webcam.

The system captures a frame from the webcam.

It compares this frame against a folder of "verification" images.

If the verification threshold is met (e.g., >0.5 detection confidence and >0.5 verification accuracy), access is granted.

📄 References
Based on the paper: Siamese Neural Networks for One-shot Image Recognition by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov.

Dataset: Labelled Faces in the Wild (LFW)
