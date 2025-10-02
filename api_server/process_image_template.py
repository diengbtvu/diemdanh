"""
Template for process_image function với Smart Mode
Copy function này vào các servers và thay MODEL_NAME
"""

def process_image_smart_template(image_data, model, mtcnn, CLASS_NAMES, 
                                 DEVICE, IMG_SIZE, CONF_THRESHOLD, 
                                 NORMALIZE_MEAN, NORMALIZE_STD,
                                 MODEL_NAME="Model"):
    """
    SMART PROCESSING TEMPLATE
    - Auto-detect: aligned face vs full scene
    - No double-crop issue
    - Top-3 predictions cho aligned faces
    """
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    import base64
    import io
    from api_server.smart_detector import smart_face_detection
    from api_server.face_preprocessing import preprocess_face_advanced
    
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
        
        # SMART DETECTION
        detection_result = smart_face_detection(frame, mtcnn)
        mode = detection_result['mode']
        print(f"[MODE] {mode.upper()}")
        
        results = []
        
        # MODE 1: ALIGNED FACE
        if mode == 'aligned':
            print(f"[{MODEL_NAME}] ALIGNED FACE - Direct prediction")
            
            # Preprocess
            face_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_tensor = torch.from_numpy(face_rgb).float().permute(2, 0, 1) / 255.0
            for i in range(3):
                face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top3_conf, top3_idx = torch.topk(probs, k=min(3, len(CLASS_NAMES)), dim=1)
                
                conf = top3_conf[0, 0].item()
                pred_class = CLASS_NAMES[top3_idx[0, 0].item()]
                
                print(f"[TOP-1] {pred_class} ({conf:.4f})")
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
                            {"class": CLASS_NAMES[top3_idx[0, i].item()], 
                             "confidence": round(top3_conf[0, i].item(), 4)}
                            for i in range(min(3, len(CLASS_NAMES)))
                        ],
                        "bounding_box": {"x1": 0, "y1": 0, "x2": w, "y2": h}
                    })
        
        # MODE 2: FULL SCENE
        else:
            print(f"[{MODEL_NAME}] FULL SCENE - MTCNN + Advanced preprocessing")
            
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
                is_good_quality = preprocess_result['is_good_quality']
                
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    conf = conf.item()
                    pred_class = CLASS_NAMES[pred_idx.item()]
                    
                    print(f"[FACE {i+1}] {pred_class} ({conf:.4f})")
                    
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
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

