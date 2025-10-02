from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN
import base64
from PIL import Image
import io
import sys
import os

# Thêm path để import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AttentionCNN
from api_server.utils import load_class_names

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

# Khởi tạo Flask-RESTX API với Swagger
api = Api(
    app,
    version='1.0',
    title='Face Recognition API - Attention CNN',
    description='API nhận dạng khuôn mặt sử dụng Attention CNN và MTCNN',
    doc='/swagger/',
    prefix='/api/v1'
)

# -------- CONFIG --------
MODEL_PATH = "../face_detection_results/face_detection_results/attention_cnn_best_model.pth"
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalization constants
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Load class names từ file
CLASS_NAMES = load_class_names(
    class_names_file=os.path.join(os.path.dirname(__file__), "class_names.txt"),
    num_classes=294
)

# Load model Attention CNN
print("[INFO] Loading Attention CNN model...")
model = AttentionCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("[INFO] Attention CNN model loaded successfully!")

# Khởi tạo MTCNN
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Namespace
ns = api.namespace('face-recognition', description='Face Recognition Operations')

# Swagger models
bounding_box_model = api.model('BoundingBox', {
    'x1': fields.Integer(required=True),
    'y1': fields.Integer(required=True),
    'x2': fields.Integer(required=True),
    'y2': fields.Integer(required=True)
})

detection_model = api.model('Detection', {
    'face_id': fields.Integer(required=True),
    'class': fields.String(required=True),
    'confidence': fields.Float(required=True),
    'bounding_box': fields.Nested(bounding_box_model, required=True)
})

prediction_response_model = api.model('PredictionResponse', {
    'success': fields.Boolean(required=True),
    'total_faces': fields.Integer(required=True),
    'detections': fields.List(fields.Nested(detection_model), required=True)
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True),
    'error': fields.String(required=True)
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(required=True),
    'model': fields.String(required=True),
    'model_loaded': fields.Boolean(required=True),
    'device': fields.String(required=True)
})

base64_input_model = api.model('Base64Input', {
    'image': fields.String(required=True)
})

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True)

def preprocess_face(face_img):
    """Preprocess face image giống như training"""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).float()
    face_tensor = face_tensor.permute(2, 0, 1)
    face_tensor = face_tensor / 255.0
    
    for i in range(3):
        face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
    
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor

def process_image(image_data):
    """Xử lý ảnh và nhận dạng khuôn mặt"""
    try:
        print(f"[DEBUG] Processing image, type: {type(image_data)}")
        
        if isinstance(image_data, str):
            print("[DEBUG] Processing base64 image")
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            print("[DEBUG] Processing uploaded file")
            image = Image.open(image_data)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        print(f"[DEBUG] Image shape: {frame.shape}")
        results = []
        
        print("[DEBUG] Detecting faces with MTCNN...")
        boxes, _ = mtcnn.detect(frame)
        print(f"[DEBUG] MTCNN detected boxes: {boxes}")

        if boxes is not None:
            print(f"[DEBUG] Found {len(boxes)} face(s)")
            for i, box in enumerate(boxes):
                print(f"[DEBUG] Processing face {i+1}: {box}")
                x1, y1, x2, y2 = [int(b) for b in box]

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_tensor = preprocess_face(face).to(DEVICE)
                
                print(f"[DEBUG] Running Attention CNN prediction for face {i+1}")
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    conf = confidence.item()
                    pred_class = CLASS_NAMES[predicted_idx.item()]
                    
                    print(f"[DEBUG] Prediction: {pred_class} with confidence {conf}")
                    
                    if conf >= CONF_THRESHOLD:
                        results.append({
                            "face_id": i + 1,
                            "class": pred_class,
                            "confidence": round(conf, 4),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            }
                        })
        else:
            print("[DEBUG] No faces detected by MTCNN")

        return {
            "success": True,
            "total_faces": len(results),
            "detections": results
        }

    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@ns.route('/predict/file')
class PredictFile(Resource):
    @ns.doc('predict_from_file')
    @ns.expect(upload_parser)
    @ns.response(200, 'Success', prediction_response_model)
    @ns.response(400, 'Bad Request', error_response_model)
    @ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """Upload an image file for face recognition"""
        try:
            args = upload_parser.parse_args()
            file = args['image']
            
            if file.filename == '':
                return {"success": False, "error": "No file selected"}, 400
            
            result = process_image(file)
            
            if result.get('success', False):
                return {
                    "success": True,
                    "total_faces": result.get('total_faces', 0),
                    "detections": result.get('detections', [])
                }, 200
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }, 500
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

@ns.route('/predict/base64')
class PredictBase64(Resource):
    @ns.doc('predict_from_base64')
    @ns.expect(base64_input_model)
    @ns.response(200, 'Success', prediction_response_model)
    @ns.response(400, 'Bad Request', error_response_model)
    @ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """Send base64 encoded image for face recognition"""
        try:
            data = request.json
            if not data or 'image' not in data:
                return {"success": False, "error": "No image data provided"}, 400
            
            result = process_image(data['image'])
            
            if result.get('success', False):
                return {
                    "success": True,
                    "total_faces": result.get('total_faces', 0),
                    "detections": result.get('detections', [])
                }, 200
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }, 500
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check')
    @ns.response(200, 'Success', health_response_model)
    def get(self):
        """Check API health status"""
        return {
            "status": "healthy",
            "model": "Attention CNN",
            "model_loaded": True,
            "device": DEVICE
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Legacy endpoint"""
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            return jsonify(process_image(file))
        elif request.json and 'image' in request.json:
            return jsonify(process_image(request.json['image']))
        else:
            return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Legacy health check"""
    return jsonify({
        "status": "healthy",
        "model": "Attention CNN",
        "model_loaded": True,
        "device": DEVICE
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Face Recognition API - Attention CNN",
        "model": "Attention CNN (Compact: 5.6MB) - 100% Test Accuracy",
        "swagger_documentation": "/swagger/",
        "api_endpoints": {
            "/api/v1/face-recognition/predict/file": "POST - Upload image file",
            "/api/v1/face-recognition/predict/base64": "POST - Send base64 image",
            "/api/v1/face-recognition/health": "GET - Health check"
        }
    })

if __name__ == '__main__':
    print(f"[INFO] ====================================")
    print(f"[INFO] Attention CNN Face Recognition API")
    print(f"[INFO] ====================================")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model: Attention CNN (Test Acc: 100%, Size: 5.6MB)")
    print(f"[INFO] Classes: {len(CLASS_NAMES)}")
    print(f"[INFO] Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("[INFO] Available endpoints:")
    print("  - Swagger: http://localhost:5003/swagger/")
    print("  - POST /api/v1/face-recognition/predict/file")
    print("  - POST /api/v1/face-recognition/predict/base64")
    print("  - GET /api/v1/face-recognition/health")
    app.run(debug=True, host='0.0.0.0', port=5003)

