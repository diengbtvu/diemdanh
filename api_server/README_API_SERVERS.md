# 🚀 Face Recognition API Servers

Tất cả các API servers sử dụng models đã train với **100% test accuracy**.

## 📦 Các Servers Available

| Server | Model | Port | Model Size | Test Acc | Best For |
|--------|-------|------|------------|----------|----------|
| `server_vanilla_cnn.py` | Vanilla CNN | 5001 | 26.8 MB | 100% | Balance |
| `server_resnet50.py` | ResNet50 | 5002 | 92.2 MB | 100% | Highest Acc |
| `server_attention_cnn.py` | Attention CNN | 5003 | 5.6 MB | 100% | **Mobile/Edge** ✅ |
| `server_yolo.py` | YOLOv8 | 5000 | - | - | Legacy |

---

## 🎯 KHUYẾN NGHỊ

### 🏆 Production Deployment
**→ Dùng `server_attention_cnn.py` (Port 5003)**

**Lý do:**
- ✅ Model nhỏ nhất (5.6 MB)
- ✅ 100% test accuracy
- ✅ Fast inference
- ✅ Perfect cho mobile/edge devices

---

## 🔧 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install flask flask-restx opencv-python torch torchvision facenet-pytorch pillow numpy
```

### 2. Kiểm tra models đã có

Đảm bảo các model files tồn tại:
```
face_detection_results/face_detection_results/
├── vanilla_cnn_best_model.pth
├── resnet50_best_model.pth
└── attention_cnn_best_model.pth
```

---

## 🚀 Chạy Servers

### Chạy từng server riêng lẻ:

**Vanilla CNN (Port 5001):**
```bash
cd api_server
python server_vanilla_cnn.py
```

**ResNet50 (Port 5002):**
```bash
cd api_server
python server_resnet50.py
```

**Attention CNN (Port 5003) - KHUYẾN NGHỊ:**
```bash
cd api_server
python server_attention_cnn.py
```

### Chạy tất cả cùng lúc (Windows):

Tạo file `run_all_servers.bat`:
```batch
@echo off
start cmd /k "python server_vanilla_cnn.py"
start cmd /k "python server_resnet50.py"
start cmd /k "python server_attention_cnn.py"
echo All servers started!
```

### Chạy tất cả cùng lúc (Linux/Mac):

Tạo file `run_all_servers.sh`:
```bash
#!/bin/bash
python server_vanilla_cnn.py &
python server_resnet50.py &
python server_attention_cnn.py &
echo "All servers started!"
```

---

## 📡 API Endpoints

**TẤT CẢ servers đều có cùng endpoints:**

### 1. Swagger Documentation
```
GET http://localhost:{PORT}/swagger/
```

### 2. Upload File
```bash
POST http://localhost:{PORT}/api/v1/face-recognition/predict/file

# Example with curl:
curl -X POST http://localhost:5003/api/v1/face-recognition/predict/file \
  -F "image=@path/to/image.jpg"
```

### 3. Base64 Image
```bash
POST http://localhost:{PORT}/api/v1/face-recognition/predict/base64

# Example:
curl -X POST http://localhost:5003/api/v1/face-recognition/predict/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_STRING_HERE"}'
```

### 4. Health Check
```bash
GET http://localhost:{PORT}/api/v1/face-recognition/health
```

### 5. Legacy Endpoints (Backward Compatibility)
```bash
POST http://localhost:{PORT}/predict
GET http://localhost:{PORT}/health
```

---

## 📤 Response Format

**Success Response:**
```json
{
  "success": true,
  "total_faces": 2,
  "detections": [
    {
      "face_id": 1,
      "class": "student_001",
      "confidence": 0.9876,
      "bounding_box": {
        "x1": 100,
        "y1": 50,
        "x2": 250,
        "y2": 200
      }
    },
    {
      "face_id": 2,
      "class": "student_042",
      "confidence": 0.9523,
      "bounding_box": {
        "x1": 300,
        "y1": 80,
        "x2": 450,
        "y2": 230
      }
    }
  ]
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message here"
}
```

---

## 🧪 Testing với Python

```python
import requests

# Test với file upload
url = "http://localhost:5003/api/v1/face-recognition/predict/file"
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# Test với base64
import base64
with open('test_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:5003/api/v1/face-recognition/predict/base64"
data = {'image': image_base64}
response = requests.post(url, json=data)
print(response.json())
```

---

## 🧪 Testing với JavaScript (Frontend)

```javascript
// Upload file
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5003/api/v1/face-recognition/predict/file', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Base64
const base64Image = '...'; // Your base64 string
fetch('http://localhost:5003/api/v1/face-recognition/predict/base64', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: base64Image })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 🔧 Configuration

Mỗi server có thể config trong file:

```python
# -------- CONFIG --------
MODEL_PATH = "../face_detection_results/..."  # Path to model
IMG_SIZE = 224                                 # Image size
CONF_THRESHOLD = 0.5                           # Confidence threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 🎯 So sánh Performance

| Model | Inference Time | Model Size | Accuracy | Recommended Use |
|-------|----------------|------------|----------|-----------------|
| Vanilla CNN | Fast | 26.8 MB | 100% | General purpose |
| ResNet50 | Medium | 92.2 MB | 100% | Highest accuracy |
| **Attention CNN** | **Fast** | **5.6 MB** | **100%** | **Mobile/Production** ✅ |

---

## 📝 Notes

1. **MTCNN** được dùng cho face detection (detect bounding boxes)
2. **PyTorch models** được dùng cho face recognition (classify người)
3. **Image preprocessing** giống hệt training:
   - Resize: 224x224
   - Normalize: ImageNet mean/std
   - Color: RGB

4. **Confidence threshold**: 0.5 (có thể thay đổi)

5. **Class names**: student_000 đến student_293 (294 classes)

---

## 🐛 Troubleshooting

### Lỗi: Model file not found
```bash
# Kiểm tra path đến model
ls ../face_detection_results/face_detection_results/

# Hoặc update MODEL_PATH trong server
MODEL_PATH = "path/to/your/model.pth"
```

### Lỗi: CUDA out of memory
```python
# Đổi sang CPU
DEVICE = "cpu"
```

### Lỗi: facenet_pytorch not found
```bash
pip install facenet-pytorch
```

### Lỗi: Port already in use
```bash
# Đổi port trong code hoặc kill process cũ
# Windows:
netstat -ano | findstr :5003
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5003
kill -9 <PID>
```

---

## 🚀 Production Deployment

### Với Gunicorn (Linux):

```bash
pip install gunicorn

# Chạy Attention CNN server
gunicorn -w 4 -b 0.0.0.0:5003 server_attention_cnn:app
```

### Với Docker:

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5003
CMD ["python", "server_attention_cnn.py"]
```

---

## 📊 Monitoring

Kiểm tra health:
```bash
curl http://localhost:5003/api/v1/face-recognition/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Attention CNN",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## 🎉 KẾT LUẬN

**Server KHUYẾN NGHỊ cho Production:**
```bash
python server_attention_cnn.py  # Port 5003
```

**Lý do:**
- ✅ 100% test accuracy
- ✅ Nhỏ nhất: 5.6 MB
- ✅ Nhanh
- ✅ Perfect cho deployment

**Tất cả endpoints tương thích với frontend hiện tại!** 🎯

