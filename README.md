🧠 AI-Based Deepfake Detection System

Detecting AI-generated images & videos with Explainable Deep Learning

An end-to-end Deepfake Detection System that identifies whether an image or video is REAL or AI-GENERATED using a fine-tuned EfficientNet-B0 model.
The system also provides confidence scores and visual explanations to improve trust and transparency.

🚀 Live Demo

🔗 Hugging Face Space:
👉 https://huggingface.co/spaces/Thanusha22/ai-deepfake-detection

✨ Key Features

✅ Image Deepfake Detection
✅ Video Deepfake Detection (Frame-wise Analysis)
✅ Confidence Score (REAL vs FAKE Probability)
✅ UNCERTAIN Class for Borderline Cases
✅ Explainable AI using Grad-CAM
✅ Optimized for Fast Inference
✅ Cloud-Deployed (Hugging Face Spaces)

🧠 Model Architecture

Base Model: EfficientNet-B0

Framework: PyTorch

Classifier: Softmax (2-Class: REAL / FAKE)

Explainability: Grad-CAM Heatmaps

Decision Logic: Probability Thresholding

The model focuses on subtle facial artifacts, texture inconsistencies, and unnatural patterns commonly found in AI-generated media.

🎯 How It Works

Upload an Image or Video

Preprocessing (Resize, Normalize, Tensor Conversion)

Model Inference using EfficientNet-B0

Confidence Score Calculation

Grad-CAM Visualization (for Images)

Final Prediction: REAL / FAKE / UNCERTAIN

🛠 Tech Stack
Category	Tools
Language	Python
Deep Learning	PyTorch
Model	EfficientNet-B0
Image Processing	OpenCV
Explainability	Grad-CAM
Web Interface	Gradio
Deployment	Hugging Face Spaces
📸 Screenshots
🔹 Image Deepfake Detection
<img width="1366" height="727" alt="Screenshot (217)" src="https://github.com/user-attachments/assets/5a35034c-672d-428f-9512-55ef66f70b5c" />
🔹 Confidence & Probability Visualization
<img width="1366" height="728" alt="Screenshot (218)" src="https://github.com/user-attachments/assets/c9f98ae3-bb12-49b5-93c7-e8c2cf296c23" />
⚠️ Limitations

Performance depends on dataset diversity

Real-world photos with heavy filters may cause uncertainty

Video analysis is frame-based (not temporal modeling)

🔮 Future Enhancements

🔹 Temporal Deepfake Detection (CNN + LSTM / Transformers)

🔹 Face Region Localization

🔹 Support for Audio Deepfakes

🔹 Mobile-friendly Interface

🔹 Improved Dataset Generalization

👩‍💻 Author

Thanusha
🎓 MCA | AI & Deep Learning Enthusiast
💡 Interested in Explainable AI & Cybersecurity

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork or contribute!
