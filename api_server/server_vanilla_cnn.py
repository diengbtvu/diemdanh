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
from models import VanillaCNN
from api_server.utils import load_class_names
from api_server.face_preprocessing import preprocess_face_advanced

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

# Khởi tạo Flask-RESTX API với Swagger
api = Api(
    app,
    version='1.0',
    title='Face Recognition API - Vanilla CNN',
    description='API nhận dạng khuôn mặt sử dụng Vanilla CNN và MTCNN',
    doc='/swagger/',
    prefix='/api/v1'
)

# -------- CONFIG --------
MODEL_PATH = "../face_detection_results/face_detection_results/vanilla_cnn_best_model.pth"
IMG_SIZE = 224              # Đúng với training
CONF_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalization constants (giống training)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Load class names từ file
CLASS_NAMES = load_class_names(
    class_names_file=os.path.join(os.path.dirname(__file__), "class_names.txt"),
    num_classes=294
)

# Load model Vanilla CNN
print("[INFO] Loading Vanilla CNN model...")
model = VanillaCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("[INFO] Vanilla CNN model loaded successfully!")

# Khởi tạo MTCNN để phát hiện khuôn mặt
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Tạo namespace cho API
ns = api.namespace('face-recognition', description='Face Recognition Operations')

# Định nghĩa models cho Swagger documentation
bounding_box_model = api.model('BoundingBox', {
    'x1': fields.Integer(required=True, description='X coordinate of top-left corner'),
    'y1': fields.Integer(required=True, description='Y coordinate of top-left corner'),
    'x2': fields.Integer(required=True, description='X coordinate of bottom-right corner'),
    'y2': fields.Integer(required=True, description='Y coordinate of bottom-right corner')
})

detection_model = api.model('Detection', {
    'face_id': fields.Integer(required=True, description='Face ID'),
    'class': fields.String(required=True, description='Predicted person name'),
    'confidence': fields.Float(required=True, description='Model confidence (0-1)'),
    'quality_score': fields.Float(required=True, description='Face image quality score (0-1)'),
    'bounding_box': fields.Nested(bounding_box_model, required=True, description='Face bounding box')
})

prediction_response_model = api.model('PredictionResponse', {
    'success': fields.Boolean(required=True, description='Whether the prediction was successful'),
    'total_faces': fields.Integer(required=True, description='Total number of faces detected'),
    'detections': fields.List(fields.Nested(detection_model), required=True, description='List of face detections')
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Always false for errors'),
    'error': fields.String(required=True, description='Error message')
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(required=True, description='Health status'),
    'model': fields.String(required=True, description='Model name'),
    'model_loaded': fields.Boolean(required=True, description='Whether the model is loaded'),
    'device': fields.String(required=True, description='Device being used (cpu/cuda)')
})

base64_input_model = api.model('Base64Input', {
    'image': fields.String(required=True, description='Base64 encoded image data')
})

# Tạo file upload parser
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True, help='Image file to process')

def preprocess_face(face_img):
    """Preprocess face image giống như training"""
    # Resize to 224x224
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor và normalize
    face_tensor = torch.from_numpy(face_rgb).float()
    face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
    face_tensor = face_tensor / 255.0  # Scale to [0, 1]
    
    # Normalize với ImageNet stats
    for i in range(3):
        face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
    
    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(0)
    
    return face_tensor

def process_image(image_data):
    """Xử lý ảnh và nhận dạng khuôn mặt"""
    try:
        print(f"[DEBUG] Processing image, type: {type(image_data)}")
        
        # Convert to numpy array
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
        
        # Detect khuôn mặt với MTCNN
        print("[DEBUG] Detecting faces with MTCNN...")
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
        print(f"[DEBUG] MTCNN detected boxes: {boxes}")

        if boxes is not None:
            print(f"[DEBUG] Found {len(boxes)} face(s)")
            for i, box in enumerate(boxes):
                print(f"[DEBUG] Processing face {i+1}: {box}")
                
                # Get landmarks for this face
                face_landmarks = landmarks[i] if landmarks is not None else None
                
                # ADVANCED PREPROCESSING
                preprocess_result = preprocess_face_advanced(
                    frame=frame,
                    bbox=box,
                    landmarks=face_landmarks,
                    enhance=True,      # Cải thiện chất lượng ảnh
                    align=True,        # Căn chỉnh khuôn mặt
                    quality_check=True # Kiểm tra chất lượng
                )
                
                if preprocess_result is None:
                    print(f"[DEBUG] Face {i+1} preprocessing failed")
                    continue
                
                face_tensor = preprocess_result['tensor'].to(DEVICE)
                quality_score = preprocess_result['quality_score']
                is_good_quality = preprocess_result['is_good_quality']
                
                print(f"[DEBUG] Face {i+1} quality: {quality_score:.2f} "
                      f"({'Good' if is_good_quality else 'Low'})")
                
                # Predict
                print(f"[DEBUG] Running Vanilla CNN prediction for face {i+1}")
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    conf = confidence.item()
                    pred_class = CLASS_NAMES[predicted_idx.item()]
                    
                    print(f"[DEBUG] Prediction: {pred_class} with confidence {conf}")
                    
                    # Adaptive threshold
                    adjusted_threshold = CONF_THRESHOLD
                    if not is_good_quality:
                        adjusted_threshold = CONF_THRESHOLD + 0.1
                    
                    if conf >= adjusted_threshold:
                        x1, y1, x2, y2 = preprocess_result['original_bbox']
                        
                        results.append({
                            "face_id": i + 1,
                            "class": pred_class,
                            "confidence": round(conf, 4),
                            "quality_score": round(quality_score, 2),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            }
                        })
        else:
            print("[DEBUG] No faces detected by MTCNN")

        print(f"[DEBUG] Final results: {len(results)} detections")
        return {
            "success": True,
            "total_faces": len(results),
            "detections": results
        }

    except Exception as e:
        print(f"[ERROR] Exception in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

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
            
            print(f"[DEBUG] Processing uploaded file: {file.filename}")
            result = process_image(file)
            print(f"[DEBUG] Process result: {result}")
            
            if result.get('success', False):
                response_data = {
                    "success": True,
                    "total_faces": result.get('total_faces', 0),
                    "detections": result.get('detections', [])
                }
                print(f"[DEBUG] Final response: {response_data}")
                return response_data, 200
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }, 500

        except Exception as e:
            print(f"[ERROR] Exception in PredictFile.post: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }, 500

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
            
            image_data = data['image']
            print(f"[DEBUG] Processing base64 image, length: {len(image_data)}")
            result = process_image(image_data)
            print(f"[DEBUG] Process result: {result}")
            
            if result.get('success', False):
                response_data = {
                    "success": True,
                    "total_faces": result.get('total_faces', 0),
                    "detections": result.get('detections', [])
                }
                print(f"[DEBUG] Final response: {response_data}")
                return response_data, 200
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }, 500

        except Exception as e:
            print(f"[ERROR] Exception in PredictBase64.post: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }, 500

@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check')
    @ns.response(200, 'Success', health_response_model)
    def get(self):
        """Check API health status"""
        return {
            "status": "healthy",
            "model": "Vanilla CNN",
            "model_loaded": True,
            "device": DEVICE
        }

# Legacy endpoints để backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để nhận dạng khuôn mặt từ ảnh (Legacy endpoint)"""
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            result = process_image(file)
            return jsonify(result)
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            result = process_image(image_data)
            return jsonify(result)
        else:
            return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (Legacy)"""
    return jsonify({
        "status": "healthy",
        "model": "Vanilla CNN",
        "model_loaded": True,
        "device": DEVICE
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint với hướng dẫn sử dụng"""
    return jsonify({
        "message": "Face Recognition API - Vanilla CNN",
        "model": "Vanilla CNN (100% Test Accuracy)",
        "swagger_documentation": "/swagger/",
        "api_endpoints": {
            "/api/v1/face-recognition/predict/file": "POST - Upload image file for face recognition",
            "/api/v1/face-recognition/predict/base64": "POST - Send base64 image for face recognition",
            "/api/v1/face-recognition/health": "GET - Check API health"
        },
        "legacy_endpoints": {
            "/predict": "POST - Upload image for face recognition (legacy)",
            "/health": "GET - Check API health (legacy)"
        }
    })

if __name__ == '__main__':
    print(f"[INFO] ====================================")
    print(f"[INFO] Vanilla CNN Face Recognition API")
    print(f"[INFO] ====================================")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model: Vanilla CNN (Test Acc: 100%)")
    print(f"[INFO] Classes: {len(CLASS_NAMES)}")
    print(f"[INFO] Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("[INFO] Available endpoints:")
    print("  - Swagger: http://localhost:5001/swagger/")
    print("  - POST /api/v1/face-recognition/predict/file")
    print("  - POST /api/v1/face-recognition/predict/base64")
    print("  - GET /api/v1/face-recognition/health")
    app.run(debug=True, host='0.0.0.0', port=5001)

