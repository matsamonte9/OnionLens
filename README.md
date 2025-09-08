🚀 Installation & Setup

  1.) CLONE THE REPO
  2.) DOWNLOAD THE MODELS
      // YOU CAN ALSO DOWNLOAD THE CHECKPOINT (machine learning code .ipynb) FOR CODE REFERENCE
      GOOGLE DRIVE LINK: https://drive.google.com/drive/folders/12lVMkWPzjWBbNGj-nQYWfTSp-tm69JnP?usp=sharing

🧅 OnionLens: Detecting Armyworm Infestation in Onions using Image Processing Techniques

  OnionLens is a machine learning–powered web application designed to assist farmers and researchers in 
  detecting armyworm infestations in onion crops. Using image processing and deep learning (CNN), the 
  system can distinguish between healthy and unhealthy onion leaves, providing farmers with faster decision-
  making tools for pest management.

📌 Features

  ✅ Onion Validation – ensures the uploaded image is a valid onion leaf before processing.
  ✅ Armyworm Detection – classifies onion leaves as Healthy or Unhealthy (infested).
  ✅ Saliency Maps (Explainable AI) – highlights infected regions for better model interpretability.
  ✅ Countermeasures & Recommendations – suggests practical agricultural practices when infestation is detected.
  ✅ User-Friendly Web App – built with Flask, integrated with a simple frontend for farmers and researchers.

🛠️ Tech Stack

  * Backend: Python, Flask
  * Deep Learning: TensorFlow / Keras (CNN models)
  * Image Processing: PIL (Pillow), Matplotlib
  * Frontend: HTML, CSS, Bootstrap, JavaScript
  * Storage: Local file system for uploads & processed images

📂 Project Structure

  OnionLens/
  │── app.py                  # Main Flask application
  │── templates/              # HTML templates (Jinja2)
  │── static/                 # CSS, JS, images
  │── images/
  │   ├── upload/             # Uploaded user images
  │   ├── processed/          # Processed images with labels & saliency maps
  │   ├── upload/healthy/     # Classified healthy onions
  │   └── upload/unhealthy/   # Classified unhealthy onions
  │── models/
  │   ├── best_onion_model.keras
  │   └── best_armyworm_model.h5
  │── README.md               # Project documentation

📊 Dataset

  Classes:
    healthy – Onion leaves without infestation
    unhealthy – Onion leaves with visible armyworm damage
    Dataset Size: ~1,038 images
    Images were split into training, validation, and test sets for model development.

🤖 Model Details

  Onion Detection Model: Validates if the input is an onion leaf.
  Armyworm Detection Model: CNN classifier with Softmax output (Healthy vs Unhealthy).
  Input Size: (256 × 256 × 3) images
  Training Techniques: Data augmentation, normalization, learning rate scheduling, cross-validation.

📄 USER GUIDE

  - For detailed instructions on **accepted file formats**, **upload limits**, and **how to use the system**, please clone the repo.

👥 Authors 

  GITHUB USERNAME: matsamonte9 – Machine Learning/Model Development (Onion Validation & Armyworm Detection)
  GITHUB USERNAME: Luwisiii - Frontend and API Integration
  

