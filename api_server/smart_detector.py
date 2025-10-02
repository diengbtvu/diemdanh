"""
Smart Face Detection Module
Tự động nhận diện xem input là aligned face hay full scene
"""

import cv2
import numpy as np


def is_aligned_face(image, min_face_ratio=0.6):
    """
    Kiểm tra xem ảnh có phải là aligned face (đã crop) hay không
    
    Logic:
    - Aligned face: Khuôn mặt chiếm 60%+ diện tích ảnh
    - Full scene: Khuôn mặt chiếm <60% hoặc nhiều objects
    
    Args:
        image: BGR image
        min_face_ratio: Tỷ lệ tối thiểu để coi là aligned face
    
    Returns:
        (is_aligned: bool, confidence: float, reason: str)
    """
    h, w = image.shape[:2]
    total_area = h * w
    
    # 1. CHECK SIZE - Aligned faces thường nhỏ (160-300px)
    if w <= 300 and h <= 300:
        # Ảnh nhỏ, có thể là aligned face
        size_score = 1.0
    elif w <= 640 and h <= 640:
        # Ảnh trung bình
        size_score = 0.5
    else:
        # Ảnh lớn, chắc là full scene
        size_score = 0.0
    
    # 2. CHECK ASPECT RATIO - Aligned faces gần vuông (0.8-1.2)
    aspect_ratio = w / h
    if 0.8 <= aspect_ratio <= 1.2:
        aspect_score = 1.0
    elif 0.6 <= aspect_ratio <= 1.5:
        aspect_score = 0.5
    else:
        aspect_score = 0.0
    
    # 3. CHECK CENTER CONCENTRATION - Face ở giữa
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lấy center region (50% giữa)
    center_h = int(h * 0.25), int(h * 0.75)
    center_w = int(w * 0.25), int(w * 0.75)
    center_region = gray[center_h[0]:center_h[1], center_w[0]:center_w[1]]
    
    # So sánh brightness center vs edges
    center_brightness = np.mean(center_region)
    edge_top = np.mean(gray[0:int(h*0.1), :])
    edge_bottom = np.mean(gray[int(h*0.9):, :])
    edge_brightness = (edge_top + edge_bottom) / 2
    
    # Aligned face: center sáng hơn edges (face được chiếu sáng)
    if center_brightness > edge_brightness + 10:
        center_score = 1.0
    elif center_brightness > edge_brightness - 10:
        center_score = 0.5
    else:
        center_score = 0.0
    
    # 4. COMBINED SCORE
    total_score = (size_score * 0.5 + aspect_score * 0.3 + center_score * 0.2)
    
    is_aligned = total_score >= 0.6
    
    # Reason
    if is_aligned:
        reason = f"Small size ({w}x{h}), aspect ratio {aspect_ratio:.2f}, face centered"
    else:
        reason = f"Large size ({w}x{h}) or complex scene"
    
    return is_aligned, total_score, reason


def smart_face_detection(image, mtcnn, face_ratio_threshold=0.6):
    """
    Smart detection: Tự động chọn mode phù hợp
    
    Args:
        image: BGR image
        mtcnn: MTCNN detector instance
        face_ratio_threshold: Threshold để coi là aligned face
    
    Returns:
        {
            'mode': 'aligned' hoặc 'scene',
            'faces': List of face regions hoặc full image,
            'boxes': Bounding boxes (nếu mode='scene'),
            'landmarks': Landmarks (nếu mode='scene'),
            'confidence': Detection confidence
        }
    """
    h, w = image.shape[:2]
    
    # Kiểm tra xem có phải aligned face không
    is_aligned, align_score, reason = is_aligned_face(image, face_ratio_threshold)
    
    print(f"[SMART DETECT] Image size: {w}x{h}")
    print(f"[SMART DETECT] Aligned face score: {align_score:.2f}")
    print(f"[SMART DETECT] Reason: {reason}")
    
    if is_aligned:
        # MODE 1: ALIGNED FACE - Dùng trực tiếp, không cần MTCNN
        print(f"[SMART DETECT] Mode: ALIGNED FACE (direct prediction)")
        return {
            'mode': 'aligned',
            'faces': [image],  # Full image là face
            'boxes': [(0, 0, w, h)],  # Full image bbox
            'landmarks': [None],
            'confidence': align_score,
            'reason': reason
        }
    else:
        # MODE 2: FULL SCENE - Cần MTCNN detect faces
        print(f"[SMART DETECT] Mode: FULL SCENE (MTCNN detection)")
        
        try:
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            if boxes is not None and len(boxes) > 0:
                faces = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # Đảm bảo bbox hợp lệ
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = image[y1:y2, x1:x2]
                        faces.append(face_crop)
                
                return {
                    'mode': 'scene',
                    'faces': faces,
                    'boxes': boxes,
                    'landmarks': landmarks,
                    'confidence': 1.0,
                    'reason': f'MTCNN detected {len(faces)} face(s)'
                }
            else:
                # MTCNN không detect được face
                print(f"[SMART DETECT] No faces detected by MTCNN")
                return {
                    'mode': 'scene',
                    'faces': [],
                    'boxes': None,
                    'landmarks': None,
                    'confidence': 0.0,
                    'reason': 'No faces detected'
                }
        except Exception as e:
            print(f"[ERROR] MTCNN detection failed: {e}")
            return {
                'mode': 'scene',
                'faces': [],
                'boxes': None,
                'landmarks': None,
                'confidence': 0.0,
                'reason': f'Detection error: {str(e)}'
            }


if __name__ == "__main__":
    # Test module
    print("Smart Face Detection Module")
    print("="*60)
    print("Auto-detects:")
    print("  - Aligned face (160x160) → Direct prediction")
    print("  - Full scene (1920x1080) → MTCNN detection")
    print("="*60)

