from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

# Khởi tạo Flask-RESTX API với Swagger
api = Api(
    app,
    version='1.0',
    title='Face Recognition API',
    description='API để nhận dạng khuôn mặt sử dụng YOLOv8 và MTCNN',
    doc='/swagger/',
    prefix='/api/v1'
)

# -------- CONFIG --------
MODEL_PATH = "best.pt"   # Model YOLOv8-cls đã train
IMG_SIZE = 160           # Kích thước ảnh như dataset
CONF_THRESHOLD = 0.5     # Ngưỡng confidence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model YOLOv8-cls
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print("[INFO] Model loaded successfully!")

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
    'face_id': fields.Integer(required=True, description='Face ID in the image'),
    'class': fields.String(required=True, description='Predicted class/person name'),
    'confidence': fields.Float(required=True, description='Confidence score (0-1)'),
    'bounding_box': fields.Nested(bounding_box_model, required=True, description='Face bounding box coordinates')
})

prediction_response_model = api.model('PredictionResponse', {
    'success': fields.Boolean(required=True, description='Whether the prediction was successful'),
    'total_faces': fields.Integer(required=True, description='Total number of faces detected', allow_null=False),
    'detections': fields.List(fields.Nested(detection_model), required=True, description='List of face detections', allow_null=False)
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Always false for errors'),
    'error': fields.String(required=True, description='Error message')
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(required=True, description='Health status'),
    'model_loaded': fields.Boolean(required=True, description='Whether the model is loaded'),
    'device': fields.String(required=True, description='Device being used (cpu/cuda)')
})

base64_input_model = api.model('Base64Input', {
    'image': fields.String(required=True, description='Base64 encoded image data')
})

# Tạo file upload parser
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True, help='Image file to process')

def process_image(image_data):
    """
    Xử lý ảnh và nhận dạng khuôn mặt
    """
    try:
        print(f"[DEBUG] Processing image, type: {type(image_data)}")
        
        # Convert base64 to numpy array
        if isinstance(image_data, str):
            # Nếu là base64 string
            print("[DEBUG] Processing base64 image")
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            # Nếu là file upload
            print("[DEBUG] Processing uploaded file")
            image = Image.open(image_data)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        print(f"[DEBUG] Image shape: {frame.shape}")
        results = []
        
        # Detect khuôn mặt
        print("[DEBUG] Detecting faces with MTCNN...")
        boxes, _ = mtcnn.detect(frame)
        print(f"[DEBUG] MTCNN detected boxes: {boxes}")

        if boxes is not None:
            print(f"[DEBUG] Found {len(boxes)} face(s)")
            for i, box in enumerate(boxes):
                print(f"[DEBUG] Processing face {i+1}: {box}")
                x1, y1, x2, y2 = [int(b) for b in box]

                # Cắt và resize khuôn mặt về 160x160
                face = frame[y1:y2, x1:x2]
                print(f"[DEBUG] Face crop shape: {face.shape}")
                if face.size == 0:
                    print(f"[DEBUG] Face {i+1} has zero size, skipping")
                    continue
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                print(f"[DEBUG] Face resized to: {face_resized.shape}")

                # Dự đoán bằng model
                print(f"[DEBUG] Running model prediction for face {i+1}")
                model_results = model.predict(face_resized, conf=CONF_THRESHOLD, verbose=False, device=DEVICE)
                print(f"[DEBUG] Model results length: {len(model_results)}")
                
                if len(model_results) > 0:
                    result = model_results[0]
                    print(f"[DEBUG] Model result probs: {result.probs}")
                    if result.probs is not None:
                        top1 = result.names[result.probs.top1]
                        conf = result.probs.top1conf.item()
                        print(f"[DEBUG] Prediction: {top1} with confidence {conf}")
                        
                        results.append({
                            "face_id": i + 1,
                            "class": top1,
                            "confidence": round(conf, 4),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            }
                        })
                    else:
                        print(f"[DEBUG] No probs in result for face {i+1}")
                else:
                    print(f"[DEBUG] No model results for face {i+1}")
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
    @ns.response(200, 'Success', prediction_response_model)
    @ns.response(400, 'Bad Request', error_response_model)
    @ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """Upload an image file for face recognition (FLEXIBLE - accepts file or base64)"""
        try:
            print(f"[DEBUG] Request headers: {dict(request.headers)}")
            print(f"[DEBUG] Request content_type: {request.content_type}")
            
            # Try FILE UPLOAD first
            if 'image' in request.files:
                file = request.files['image']
                if file.filename == '':
                    return {"success": False, "error": "No file selected"}, 400
                
                print(f"[DEBUG] Processing uploaded file: {file.filename}")
                result = process_image(file)
            
            # Try BASE64 in JSON
            elif request.json and 'image' in request.json:
                print(f"[DEBUG] Processing base64 image from JSON")
                result = process_image(request.json['image'])
            
            # Try form data with base64
            elif request.form and 'image' in request.form:
                print(f"[DEBUG] Processing base64 from form data")
                result = process_image(request.form['image'])
            
            else:
                print(f"[ERROR] No image found in request")
                print(f"[DEBUG] request.files: {list(request.files.keys())}")
                print(f"[DEBUG] request.json: {request.json}")
                print(f"[DEBUG] request.form: {dict(request.form)}")
                return {
                    "success": False,
                    "error": "No image provided. Send file upload or base64 in JSON"
                }, 400
            
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
            
            # Trả về response trực tiếp không qua marshal
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
            "model_loaded": True,
            "device": DEVICE
        }

# Giữ lại endpoint cũ để backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint để nhận dạng khuôn mặt từ ảnh (Legacy endpoint)
    """
    try:
        # Kiểm tra xem có file được upload không
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            result = process_image(file)
            return jsonify(result)
        
        # Kiểm tra xem có base64 image không
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            result = process_image(image_data)
            return jsonify(result)
        
        else:
            return jsonify({"error": "No image provided"}), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint (Legacy)
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE
    })

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint với hướng dẫn sử dụng
    """
    return jsonify({
        "message": "Face Recognition API",
        "swagger_documentation": "/swagger/",
        "api_endpoints": {
            "/api/v1/face-recognition/predict/file": "POST - Upload image file for face recognition",
            "/api/v1/face-recognition/predict/base64": "POST - Send base64 image for face recognition",
            "/api/v1/face-recognition/health": "GET - Check API health"
        },
        "legacy_endpoints": {
            "/predict": "POST - Upload image for face recognition (legacy)",
            "/health": "GET - Check API health (legacy)"
        },
        "usage": {
            "file_upload": "Send POST request to /api/v1/face-recognition/predict/file with 'image' file",
            "base64": "Send POST request to /api/v1/face-recognition/predict/base64 with JSON: {'image': 'base64_string'}"
        }
    })

if __name__ == '__main__':
    print(f"[INFO] API Server running on device: {DEVICE}")
    print("[INFO] Available endpoints:")
    print("  - Swagger Documentation: http://localhost:5000/swagger/")
    print("  - POST /api/v1/face-recognition/predict/file - Face recognition (file upload)")
    print("  - POST /api/v1/face-recognition/predict/base64 - Face recognition (base64)")
    print("  - GET /api/v1/face-recognition/health - Health check")
    print("  - Legacy endpoints still available at /predict and /health")
    app.run(debug=True, host='0.0.0.0', port=5000)