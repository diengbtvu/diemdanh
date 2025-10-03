"""
Attention CNN API Server với YOLO Face Detection
Sử dụng CÙNG LOGIC crop như khi tạo dataset
→ Kết quả tốt nhất, không bị mismatch preprocessing
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import cv2
import torch
import numpy as np
import base64
from PIL import Image
import io
import sys
import os

# Import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AttentionCNN
from api_server.utils import load_class_names
from api_server.yolo_face_detector import get_yolo_face_detector

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

api = Api(
    app,
    version='3.0',
    title='Face Recognition API - Attention CNN + YOLO Detection',
    description='API sử dụng YOLO face detection (GIỐNG dataset preprocessing) + Attention CNN',
    doc='/swagger/',
    prefix='/api/v1'
)

# -------- CONFIG --------
MODEL_PATH = "../face_detection_results/face_detection_results/attention_cnn_best_model.pth"
YOLO_FACE_MODEL = 'yolov8n-face.pt'  # YOLO face detection (tự động download)
IMG_SIZE_DATASET = 160  # Size của dataset (aligned faces)
IMG_SIZE_MODEL = 224    # Size cho model (training size)
MARGIN = 32             # Margin khi crop (GIỐNG dataset!)
CONF_THRESHOLD = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Load class names
CLASS_NAMES = load_class_names(
    class_names_file=os.path.join(os.path.dirname(__file__), "class_names.txt"),
    num_classes=294
)

# Load Attention CNN model
print("[INFO] Loading Attention CNN model...")
model = AttentionCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()
print("[INFO] Attention CNN model loaded!")

# Load YOLO face detector
print("[INFO] Loading YOLO face detector...")
yolo_detector = get_yolo_face_detector(
    model_path=YOLO_FACE_MODEL,
    margin=MARGIN,
    image_size=IMG_SIZE_DATASET
)
print("[INFO] YOLO face detector loaded!")

# Namespace
ns = api.namespace('face-recognition', description='Face Recognition with YOLO Detection')

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
    'top_3_predictions': fields.List(fields.Raw),
    'bounding_box': fields.Nested(bounding_box_model, required=True)
})

prediction_response_model = api.model('PredictionResponse', {
    'success': fields.Boolean(required=True),
    'total_faces': fields.Integer(required=True),
    'detections': fields.List(fields.Nested(detection_model)),
    'detector': fields.String(description='Face detector used')
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True),
    'error': fields.String(required=True)
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(required=True),
    'model': fields.String(required=True),
    'detector': fields.String(required=True),
    'device': fields.String(required=True)
})

base64_input_model = api.model('Base64Input', {
    'image': fields.String(required=True)
})

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True)


def preprocess_aligned_face(face_pil_160):
    """
    Preprocess aligned face từ 160x160 (dataset format) → 224x224 (model input)
    
    Args:
        face_pil_160: PIL Image 160x160 (từ YOLO detector)
    
    Returns:
        Tensor (1, 3, 224, 224) ready for model
    """
    # Resize 160 → 224
    face_224 = face_pil_160.resize((IMG_SIZE_MODEL, IMG_SIZE_MODEL), resample=Image.BILINEAR)
    
    # Convert to numpy
    face_np = np.array(face_224)
    
    # Convert to tensor
    face_tensor = torch.from_numpy(face_np).float()
    face_tensor = face_tensor.permute(2, 0, 1)  # HWC -> CHW
    face_tensor = face_tensor / 255.0
    
    # Normalize với ImageNet stats
    for i in range(3):
        face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
    
    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(0)
    
    return face_tensor


def process_image(image_data):
    """
    Process image với YOLO face detection
    CÙNG LOGIC như khi tạo dataset
    """
    try:
        # Load image
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = Image.open(image_data)
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        h, w = frame_bgr.shape[:2]
        print(f"[INPUT] Image size: {w}x{h}")
        
        # Detect faces với YOLO (CÙNG LOGIC dataset)
        print("[DETECTOR] Using YOLO face detection (same as dataset preprocessing)")
        
        aligned_faces = yolo_detector.detect_and_align(
            frame_bgr,
            detect_multiple_faces=True  # Detect tất cả faces
        )
        
        if len(aligned_faces) == 0:
            print("[WARN] No faces detected")
            return {
                "success": True,
                "total_faces": 0,
                "detections": [],
                "detector": "YOLO (yolov8n-face)"
            }
        
        print(f"[DETECTED] {len(aligned_faces)} face(s) with YOLO")
        
        results = []
        
        for i, face_result in enumerate(aligned_faces):
            # Aligned face đã ở 160x160 (giống dataset)
            aligned_face_pil = face_result['aligned_face']
            
            print(f"[FACE {i+1}] Processing aligned face 160x160")
            
            # Preprocess: 160 → 224 → Normalize
            face_tensor = preprocess_aligned_face(aligned_face_pil).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Top-3 predictions
                top3_conf, top3_idx = torch.topk(probs, k=min(3, len(CLASS_NAMES)), dim=1)
                
                conf = top3_conf[0, 0].item()
                pred_idx = top3_idx[0, 0].item()
                pred_class = CLASS_NAMES[pred_idx]
                
                print(f"[PREDICT] Face {i+1}: {pred_class} (confidence: {conf:.4f})")
                print(f"[TOP-3]")
                for j in range(min(3, len(CLASS_NAMES))):
                    cls_name = CLASS_NAMES[top3_idx[0, j].item()]
                    cls_conf = top3_conf[0, j].item()
                    print(f"  {j+1}. {cls_name}: {cls_conf:.4f}")
                
                if conf >= CONF_THRESHOLD:
                    # Get original bbox
                    bbox = face_result['bbox']
                    
                    results.append({
                        "face_id": i + 1,
                        "class": pred_class,
                        "confidence": round(conf, 4),
                        "top_3_predictions": [
                            {
                                "class": CLASS_NAMES[top3_idx[0, j].item()],
                                "confidence": round(top3_conf[0, j].item(), 4)
                            }
                            for j in range(min(3, len(CLASS_NAMES)))
                        ],
                        "bounding_box": {
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3])
                        }
                    })
                else:
                    print(f"[WARN] Low confidence {conf:.4f} < threshold {CONF_THRESHOLD}")
        
        return {
            "success": True,
            "total_faces": len(results),
            "detections": results,
            "detector": "YOLO (yolov8n-face - same as dataset preprocessing)"
        }
    
    except Exception as e:
        print("="*80)
        print("[ERROR] EXCEPTION IN process_image:")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("="*80)
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


@ns.route('/predict/file')
class PredictFile(Resource):
    @ns.doc('predict_from_file')
    @ns.expect(upload_parser)
    @ns.response(200, 'Success', prediction_response_model)
    @ns.response(400, 'Bad Request', error_response_model)
    def post(self):
        """Upload image file"""
        try:
            args = upload_parser.parse_args()
            file = args['image']
            if file.filename == '':
                return {"success": False, "error": "No file selected"}, 400
            result = process_image(file)
            return result, 200 if result.get('success') else 500
        except Exception as e:
            print(f"[ERROR in endpoint] {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}, 500


@ns.route('/predict/base64')
class PredictBase64(Resource):
    @ns.doc('predict_from_base64')
    @ns.expect(base64_input_model)
    @ns.response(200, 'Success', prediction_response_model)
    def post(self):
        """Base64 encoded image"""
        try:
            data = request.json
            if not data or 'image' not in data:
                return {"success": False, "error": "No image"}, 400
            result = process_image(data['image'])
            return result, 200 if result.get('success') else 500
        except Exception as e:
            print(f"[ERROR in endpoint] {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}, 500


@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check')
    @ns.response(200, 'Success', health_response_model)
    def get(self):
        """Health check"""
        return {
            "status": "healthy",
            "model": "Attention CNN",
            "detector": "YOLO (yolov8n-face)",
            "device": DEVICE
        }


@app.route('/predict', methods=['POST'])
def predict():
    """Legacy endpoint"""
    try:
        if 'image' in request.files:
            return jsonify(process_image(request.files['image']))
        elif request.json and 'image' in request.json:
            return jsonify(process_image(request.json['image']))
        else:
            return jsonify({"error": "No image"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Legacy health"""
    return jsonify({"status": "healthy", "model": "Attention CNN + YOLO", "device": DEVICE})


@app.route('/', methods=['GET'])
def home():
    """Home"""
    return jsonify({
        "message": "Face Recognition API - Attention CNN + YOLO Detection",
        "model": "Attention CNN (100% Test Acc)",
        "detector": "YOLO (yolov8n-face) - Same as dataset preprocessing",
        "advantages": [
            "Uses SAME face detection as dataset creation",
            "No preprocessing mismatch",
            "Better results with aligned faces",
            "Consistent pipeline"
        ],
        "swagger": "/swagger/"
    })


if __name__ == '__main__':
    print("="*70)
    print("Attention CNN + YOLO Face Detection API")
    print("="*70)
    print("KEY FEATURE: Uses SAME face detection as dataset!")
    print("="*70)
    print(f"Model: Attention CNN (100% test accuracy)")
    print(f"Detector: YOLO (yolov8n-face)")
    print(f"Device: {DEVICE}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("="*70)
    print("Pipeline:")
    print("  1. YOLO detect face (margin=32, same as dataset)")
    print("  2. Crop to 160x160 (same as dataset)")
    print("  3. Resize to 224x224 (model input)")
    print("  4. Normalize & predict")
    print("="*70)
    print("Swagger: http://localhost:5005/swagger/")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5005)

