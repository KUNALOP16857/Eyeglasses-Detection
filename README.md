# 👓 Ultra-Fast Eyeglasses Detection 

A highly efficient, lightweight Deep Learning pipeline for detecting eyeglasses in real-time. 

While most face-attribute models rely on massive, heavy architectures (like ResNet or VGG), we built this project from the ground up to solve a specific problem: **running AI on mobile devices and AR platforms (like Lens Studio) without draining the battery, causing lag, or racking up massive cloud server costs.**

## ⚡ The Core Focus: Zero-Latency & Low Cost

If you are building an AR lens or a mobile app, sending images to a cloud server for inference is expensive and slow. To run this locally on the edge, we had to drastically reduce the model size. 

Here is how we optimized for speed and cost:
* **No Cloud Compute Needed:** The model is so small it runs entirely on-device. Zero server costs, zero network latency.
* **Depthwise Separable Convolutions:** Inspired by MobileNet, we split standard convolutions into two layers. This slashed our parameter count and compute requirements by a massive margin while keeping accuracy high.
* **Baked-in Normalization:** We divided the first layer weights by 255 before exporting to ONNX. This eliminates the need for extra image-preprocessing steps in the final app, saving precious milliseconds during inference.

## 🧠 Model Architecture & Training Highlights

Because we aggressively shrank the model, we had to be incredibly smart about how we trained it to ensure it didn't lose accuracy.

* **Anti-Aliasing (Shift-Invariance):** We replaced standard MaxPool with Zhang's BlurPool. If a user's face moves slightly in the camera frame, the prediction remains perfectly stable. No flickering.
* **Tackling Imbalanced Data:** The CelebA dataset only has ~6.5% glasses wearers. Instead of letting the model blindly guess "No Glasses," we used Undersampling and a Weighted Random Sampler to force it to learn actual eyewear features.
* **Consistency Regularization:** We used an MSE Consistency Loss alongside standard BCE. If an image is slightly flipped or shifted, the model is penalized if its prediction changes. 

## 📊 Performance
We evaluate success using the **F1-Score**, not just raw accuracy. 
By balancing Precision and Recall, we ensure the app doesn't trigger annoying "phantom" detections (false positives), nor does it miss glasses when they are actually there (false negatives).

## 🔮 Future Scope: The E-Commerce Engine
This detection model is the foundational trigger for an automated style and wellness engine:
1. **If Glasses are Detected (Wearer):** Trigger a smart recommendation engine to suggest trending, prescription-ready Sunglasses that match their face shape.
2. **If No Glasses Detected (Non-Wearer):** Push contextual wellness notifications suggesting "Zero-Power" Blue-Light blocking lenses to prevent screen fatigue.
3. **AR Integration:** Instantly load the recommended frames onto the user's face in Lens Studio for a seamless Virtual Try-On experience.

## 🛠️ Quick Start

**1. Install Dependencies**
```bash
pip install torch torchvision pandas tqdm pillow matplotlib

**2. Prepare the Data (The CelebA Dataset)**
Grab the img_align_celeba.zip and list_attr_celeba.txt from the official CelebA dataset. Drop them directly into your working directory. The notebook handles the rest of the parsing so you don't have to worry about manual folder structuring.

**3. Train & Export (Optimized for Edge)**
Run the provided Jupyter notebook. Because the architecture is so small, you won't need a massive, expensive cloud GPU cluster to train this. The script automatically handles the data equalization, trains the lightweight CNN, tracks the F1 metrics, and compiles everything down into a highly optimized classifier.onnx file.

**4. Deploy (Zero-Friction Integration)**
Take that .onnx file and drop it straight into Lens Studio, iOS CoreML, or your preferred edge platform. Because the weights are pre-normalized and the architecture is minimal, it runs locally with near-zero latency right out of the box.
