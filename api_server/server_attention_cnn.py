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
from api_server.face_preprocessing import preprocess_face_advanced
from api_server.smart_detector import smart_face_detection

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
CONF_THRESHOLD = 0.3  # Giảm xuống 0.3 để capture aligned faces tốt hơn
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
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
    'face_id': fields.Integer(required=True, description='Face ID'),
    'class': fields.String(required=True, description='Predicted person name'),
    'confidence': fields.Float(required=True, description='Model confidence (0-1)'),
    'quality_score': fields.Float(required=True, description='Face image quality score (0-1)'),
    'bounding_box': fields.Nested(bounding_box_model, required=True, description='Face bounding box')
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

def preprocess_face_simple(face_img):
    """Preprocess face image - Simple version (backward compatibility)"""
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
    """
    SMART PROCESSING: Tự động nhận diện và xử lý
    - Aligned face (160x160) → Direct prediction (không dùng MTCNN)
    - Full scene → MTCNN detection + Advanced preprocessing
    """
    try:
        # Load image
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = Image.open(image_data)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]
        print(f"[INPUT] Image size: {w}x{h}")
        
        # ===== SMART DETECTION =====
        detection_result = smart_face_detection(frame, mtcnn)
        mode = detection_result['mode']
        
        results = []
        
        # ===== MODE 1: ALIGNED FACE (ảnh đã crop) =====
        if mode == 'aligned':
            print("[MODE] ALIGNED FACE - Direct prediction")
            
            # Preprocess trực tiếp toàn bộ ảnh
            face_tensor = preprocess_face_simple(frame).to(DEVICE)
            
            # Predict với top-3
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top3_conf, top3_idx = torch.topk(probs, k=min(3, len(CLASS_NAMES)), dim=1)
                
                conf = top3_conf[0, 0].item()
                pred_idx = top3_idx[0, 0].item()
                pred_class = CLASS_NAMES[pred_idx]
                
                print(f"[TOP-1] {pred_class} (confidence: {conf:.4f})")
                print(f"[TOP-2] {CLASS_NAMES[top3_idx[0, 1].item()]} ({top3_conf[0, 1].item():.4f})")
                print(f"[TOP-3] {CLASS_NAMES[top3_idx[0, 2].item()]} ({top3_conf[0, 2].item():.4f})")
                
                if conf >= CONF_THRESHOLD:
                    results.append({
                        "face_id": 1,
                        "class": pred_class,
                        "confidence": round(conf, 4),
                        "quality_score": 1.0,
                        "mode": "aligned_face",
                        "top_3_predictions": [
                            {
                                "class": CLASS_NAMES[top3_idx[0, i].item()],
                                "confidence": round(top3_conf[0, i].item(), 4)
                            }
                            for i in range(min(3, len(CLASS_NAMES)))
                        ],
                        "bounding_box": {"x1": 0, "y1": 0, "x2": w, "y2": h}
                    })
                else:
                    print(f"[WARN] Low confidence {conf:.4f} - Might be unknown person")
        
        # ===== MODE 2: FULL SCENE (cần MTCNN) =====
        else:
            print("[MODE] FULL SCENE - MTCNN detection + Advanced preprocessing")
            
            faces_list = detection_result['faces']
            boxes = detection_result['boxes']
            landmarks = detection_result['landmarks']
            
            if len(faces_list) == 0:
                print("[WARN] No faces detected")
                return {
                    "success": True,
                    "total_faces": 0,
                    "detections": [],
                    "processing_mode": mode,
                    "message": "No faces detected"
                }
            
            print(f"[DETECTED] {len(faces_list)} face(s)")
            
            for i, box in enumerate(boxes):
                face_landmarks = landmarks[i] if landmarks is not None else None
                
                # Advanced preprocessing
                preprocess_result = preprocess_face_advanced(
                    frame, box, face_landmarks,
                    enhance=True, align=True, quality_check=True
                )
                
                if preprocess_result is None:
                    continue
                
                face_tensor = preprocess_result['tensor'].to(DEVICE)
                quality_score = preprocess_result['quality_score']
                is_good_quality = preprocess_result['is_good_quality']
                
                print(f"[FACE {i+1}] Quality: {quality_score:.2f}")
                
                # Predict
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    conf = conf.item()
                    pred_class = CLASS_NAMES[pred_idx.item()]
                    
                    print(f"[PREDICT] Face {i+1}: {pred_class} ({conf:.4f})")
                    
                    # Adaptive threshold
                    threshold = CONF_THRESHOLD + (0.15 if not is_good_quality else 0)
                    
                    if conf >= threshold:
                        x1, y1, x2, y2 = preprocess_result['original_bbox']
                        results.append({
                            "face_id": i + 1,
                            "class": pred_class,
                            "confidence": round(conf, 4),
                            "quality_score": round(quality_score, 2),
                            "mode": "scene_detection",
                            "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                        })

        return {
            "success": True,
            "total_faces": len(results),
            "detections": results,
            "processing_mode": mode
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

