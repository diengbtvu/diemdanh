from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN
import base64
from PIL import Image
import io
import sys
import os
import pickle

# Thêm path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_arcface import ArcFaceResNet50
from api_server.smart_detector import smart_face_detection
from api_server.face_preprocessing import preprocess_face_advanced

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

api = Api(
    app,
    version='2.0',
    title='Face Recognition API - ArcFace (Embedding-based)',
    description='API nhận dạng khuôn mặt sử dụng ArcFace embeddings - BEST GENERALIZATION',
    doc='/swagger/',
    prefix='/api/v1'
)

# -------- CONFIG --------
MODEL_PATH = "../face_detection_results/face_detection_results/arcface_resnet50_best_model.pth"
DATABASE_PATH = "../face_detection_results/face_detection_results/arcface_embedding_database.pkl"
IMG_SIZE = 224
SIMILARITY_THRESHOLD = 0.4  # Cosine similarity threshold (0.4-0.6 recommended)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Load embedding database
print("[INFO] Loading embedding database...")
try:
    with open(DATABASE_PATH, 'rb') as f:
        database = pickle.load(f)
    
    CLASS_NAMES = database['class_names']
    DATABASE_EMBEDDINGS = torch.from_numpy(database['embeddings']).float().to(DEVICE)
    
    print(f"[OK] Database loaded:")
    print(f"  - Classes: {len(CLASS_NAMES)}")
    print(f"  - Embedding size: {database['embedding_size']}")
    print(f"  - Database shape: {DATABASE_EMBEDDINGS.shape}")
    
except Exception as e:
    print(f"[ERROR] Failed to load database: {e}")
    print("\nPlease build database first:")
    print("  python build_arcface_database.py")
    exit(1)

# Load ArcFace model
print("[INFO] Loading ArcFace model...")
model = ArcFaceResNet50(num_classes=len(CLASS_NAMES), embedding_size=512, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()
print("[INFO] ArcFace model loaded successfully!")

# MTCNN
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
    'class': fields.String(required=True, description='Matched person name'),
    'similarity': fields.Float(required=True, description='Cosine similarity (0-1)'),
    'confidence': fields.Float(required=True, description='Match confidence (similarity * 100)'),
    'quality_score': fields.Float(required=True, description='Face quality (0-1)'),
    'mode': fields.String(required=True, description='Processing mode'),
    'is_known': fields.Boolean(required=True, description='Is person in database'),
    'bounding_box': fields.Nested(bounding_box_model, required=True)
})

prediction_response_model = api.model('PredictionResponse', {
    'success': fields.Boolean(required=True),
    'total_faces': fields.Integer(required=True),
    'detections': fields.List(fields.Nested(detection_model), required=True),
    'processing_mode': fields.String(required=True)
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True),
    'error': fields.String(required=True)
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(required=True),
    'model': fields.String(required=True),
    'approach': fields.String(required=True),
    'database_size': fields.Integer(required=True),
    'device': fields.String(required=True)
})

base64_input_model = api.model('Base64Input', {
    'image': fields.String(required=True)
})

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True)

def preprocess_aligned_face(face_img):
    """Preprocess aligned face"""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).float().permute(2, 0, 1) / 255.0
    
    for i in range(3):
        face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
    
    return face_tensor.unsqueeze(0)

def match_embedding(query_embedding, database_embeddings, class_names, threshold=0.4):
    """
    Match query embedding với database
    
    Args:
        query_embedding: (1, 512) normalized embedding
        database_embeddings: (num_classes, 512) normalized embeddings
        class_names: List of class names
        threshold: Similarity threshold
    
    Returns:
        {
            'class': Matched class name,
            'similarity': Cosine similarity,
            'is_known': True if similarity > threshold,
            'top_5': Top 5 matches
        }
    """
    # Compute cosine similarities
    similarities = torch.mm(query_embedding, database_embeddings.t())[0]  # (num_classes,)
    
    # Get top-5
    top5_sim, top5_idx = torch.topk(similarities, k=min(5, len(class_names)))
    
    # Best match
    best_sim = top5_sim[0].item()
    best_idx = top5_idx[0].item()
    best_class = class_names[best_idx]
    
    # Is known person?
    is_known = best_sim >= threshold
    
    # Top-5 matches
    top_5_matches = [
        {
            'class': class_names[top5_idx[i].item()],
            'similarity': round(top5_sim[i].item(), 4)
        }
        for i in range(len(top5_sim))
    ]
    
    return {
        'class': best_class if is_known else 'unknown',
        'similarity': best_sim,
        'is_known': is_known,
        'top_5': top_5_matches
    }

def process_image(image_data):
    """
    SMART PROCESSING with ArcFace Embeddings
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
        print(f"[INPUT] Image: {w}x{h}")
        
        # Smart detection
        detection_result = smart_face_detection(frame, mtcnn)
        mode = detection_result['mode']
        
        results = []
        
        # MODE 1: ALIGNED FACE
        if mode == 'aligned':
            print("[MODE] ALIGNED FACE - Direct embedding extraction")
            
            face_tensor = preprocess_aligned_face(frame).to(DEVICE)
            
            with torch.no_grad():
                # Extract embedding
                embedding = model(face_tensor)  # (1, 512)
                
                # Match với database
                match_result = match_embedding(
                    embedding, DATABASE_EMBEDDINGS, CLASS_NAMES, 
                    threshold=SIMILARITY_THRESHOLD
                )
                
                print(f"[MATCH] {match_result['class']} (similarity: {match_result['similarity']:.4f})")
                print(f"[TOP-5]")
                for i, m in enumerate(match_result['top_5'], 1):
                    print(f"  {i}. {m['class']}: {m['similarity']:.4f}")
                
                results.append({
                    "face_id": 1,
                    "class": match_result['class'],
                    "similarity": round(match_result['similarity'], 4),
                    "confidence": round(match_result['similarity'] * 100, 2),
                    "quality_score": 1.0,
                    "mode": "aligned_face",
                    "is_known": match_result['is_known'],
                    "top_5_matches": match_result['top_5'],
                    "bounding_box": {"x1": 0, "y1": 0, "x2": w, "y2": h}
                })
        
        # MODE 2: FULL SCENE
        else:
            print("[MODE] FULL SCENE - MTCNN + Embedding matching")
            
            faces_list = detection_result['faces']
            boxes = detection_result['boxes']
            landmarks = detection_result['landmarks']
            
            if len(faces_list) == 0:
                return {
                    "success": True,
                    "total_faces": 0,
                    "detections": [],
                    "processing_mode": mode
                }
            
            for i, box in enumerate(boxes):
                face_landmarks = landmarks[i] if landmarks is not None else None
                
                preprocess_result = preprocess_face_advanced(
                    frame, box, face_landmarks,
                    enhance=True, align=True, quality_check=True
                )
                
                if preprocess_result is None:
                    continue
                
                face_tensor = preprocess_result['tensor'].to(DEVICE)
                quality_score = preprocess_result['quality_score']
                
                with torch.no_grad():
                    # Extract embedding
                    embedding = model(face_tensor)
                    
                    # Match
                    match_result = match_embedding(
                        embedding, DATABASE_EMBEDDINGS, CLASS_NAMES,
                        threshold=SIMILARITY_THRESHOLD
                    )
                    
                    print(f"[FACE {i+1}] {match_result['class']} ({match_result['similarity']:.4f})")
                    
                    x1, y1, x2, y2 = preprocess_result['original_bbox']
                    results.append({
                        "face_id": i + 1,
                        "class": match_result['class'],
                        "similarity": round(match_result['similarity'], 4),
                        "confidence": round(match_result['similarity'] * 100, 2),
                        "quality_score": round(quality_score, 2),
                        "mode": "scene_detection",
                        "is_known": match_result['is_known'],
                        "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                    })

        return {
            "success": True,
            "total_faces": len(results),
            "detections": results,
            "processing_mode": mode
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@ns.route('/predict/file')
class PredictFile(Resource):
    @ns.doc('predict_from_file')
    @ns.expect(upload_parser)
    @ns.response(200, 'Success', prediction_response_model)
    def post(self):
        """Upload image file for face recognition"""
        try:
            args = upload_parser.parse_args()
            file = args['image']
            if file.filename == '':
                return {"success": False, "error": "No file selected"}, 400
            result = process_image(file)
            return result, 200 if result.get('success') else 500
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

@ns.route('/predict/base64')
class PredictBase64(Resource):
    @ns.doc('predict_from_base64')
    @ns.expect(base64_input_model)
    @ns.response(200, 'Success', prediction_response_model)
    def post(self):
        """Base64 image for face recognition"""
        try:
            data = request.json
            if not data or 'image' not in data:
                return {"success": False, "error": "No image"}, 400
            result = process_image(data['image'])
            return result, 200 if result.get('success') else 500
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check')
    @ns.response(200, 'Success', health_response_model)
    def get(self):
        """Health check"""
        return {
            "status": "healthy",
            "model": "ArcFace ResNet50",
            "approach": "Embedding-based (Metric Learning)",
            "database_size": len(CLASS_NAMES),
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
    return jsonify({
        "status": "healthy",
        "model": "ArcFace",
        "approach": "Embedding-based",
        "device": DEVICE
    })

@app.route('/', methods=['GET'])
def home():
    """Home"""
    return jsonify({
        "message": "Face Recognition API - ArcFace (Best Generalization)",
        "model": "ArcFace ResNet50",
        "approach": "Embedding-based (Metric Learning)",
        "features": [
            "Best generalization to unseen images",
            "Cosine similarity matching",
            "Top-5 similar faces",
            "Unknown person detection",
            "Smart mode (aligned vs scene)"
        ],
        "swagger": "/swagger/"
    })

if __name__ == '__main__':
    print("="*70)
    print("ArcFace Face Recognition API - BEST GENERALIZATION")
    print("="*70)
    print(f"Approach: Embedding-based (Metric Learning)")
    print(f"Device: {DEVICE}")
    print(f"Database: {len(CLASS_NAMES)} persons")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print("="*70)
    print("Advantages over Classification:")
    print("  ✓ Better generalization to unseen images")
    print("  ✓ Can detect unknown persons")
    print("  ✓ More robust to pose/lighting variations")
    print("="*70)
    print("Swagger: http://localhost:5004/swagger/")
    app.run(debug=True, host='0.0.0.0', port=5004)

