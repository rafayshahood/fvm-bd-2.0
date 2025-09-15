# Face Verification Backend (FastAPI)

This project is a FastAPI-based backend for performing face verification through a series of validation steps. It checks if a submitted ID image and video belong to the same person, while also validating video authenticity using anti-spoofing and deepfake detection.

---

## 🔧 Features
### Live ID Verifiaction
- ID presence inside Rectangle
- Face present inside ID
- Not too low/high brightness on overall frame
- Not too low/high brightness on face

### Live Face Video Verifiaction
- Face presence and single-face validation
- Glasses detection
- Anti-spoofing check (real vs. spoofed faces)
- Not too low/high brightness on overall frame
- Deepfake detection 

### Verification between Live ID and Live Face Video
- Final match Live Video with Live ID image

---

## 🚀 Setup Instructions

### Step 1: Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

---

### Step 2: Clone the repository

```bash
git clone https://github.com/rafayshahood/fvm-bd.git
cd fvm-bd
```

---

### Step 3: Create a temp directory

```bash
mkdir temp
```

---

### Step 4: Install dependencies

```bash
pip install --upgrade pip setuptools wheel

pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pillow shutilwhich ultralytics insightface mediapipe scikit-learn gdown 

pip install basicsr facexlib realesrgan

apt update && apt install -y cmake build-essential
apt update && apt install -y ffmpeg

pip install onnxruntime-gpu face-recognition==1.3.0 albumentations==1.3.0 decord==0.6.0 timm==0.6.5

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 5: Download YOLOv8 Model Weights

```bash
gdown --id 1Zryyh4zdEAHcGREQkje8ijCqktJgNzgD -O yolov8n-face-lindevs.pt
gdown --id 1clRE4pKUhU_tVVmd7F3ofxLuzjltIdRO -O iddetection.pt
gdown --id 1KtP11Novpt2XX2SEcMQOPPqWN56XLX-f -O glass-classification.pt

```

Place the downloaded file in the root directory.

---

### Step 6: Download AntiSpoof Model Weights
From the main repo folder:

```bash
cd Silent_Face_Anti_Spoofing/resources
gdown --folder https://drive.google.com/drive/folders/13KYghJSu6M6gePEKfJUAxZAn9wGt9B0B
```

---

### Step 7: Download Deepfake Model Weights
From the main repo folder:
```bash
cd GenConViT/weight

wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth
wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth
```

---

### Step 8: Download Image Enhancement Model Weights
From the main repo folder:
```bash
 mkdir -p GFPGAN/experiments/pretrained_models
 gdown 1jHlpgqZxYsbHrreKcDgG_qKYX01EuqZw -O GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth
```

---

---

### Step 9: Install Correct Numpy Version (Optional if library issue occurs)
From the main repo folder:
```bash
 pip install --force-reinstall --no-cache-dir numpy==1.26.4

```

---

### Step 10: Run the Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8888
```

---

## 📂 Project Structure

- `main.py` – Entry point of the FastAPI backend
- `id.py` – ID Frame processing, validation and ID enhancement logic
- `all_video.py` – Video Frame processing and validation logic
- `new_verif.py` – Handles final face verification
- `GenConViT/` – Deepfake detection module and weights
- `temp/` – Temporary folder for per-request processing
- `yolov8n-face-lindevs.pt` – YOLO model for face/spoof detection
- `glasses-classification.pt` – YOLO model for glasses detection

---


## 👨‍💻 Author

**Muhammad Abdul Rafay Shahood**  
[GitHub Profile](https://github.com/rafayshahood)
