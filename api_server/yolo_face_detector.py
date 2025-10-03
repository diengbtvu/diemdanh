"""
YOLO Face Detection Module
Sử dụng CÙNG LOGIC như khi tạo aligned_faces dataset
để đảm bảo kết quả prediction tốt nhất
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch


class YOLOFaceDetector:
    """
    YOLO Face Detector với CÙNG logic crop như dataset preprocessing
    """
    
    def __init__(self, model_path='yolov8n-face.pt', margin=32, image_size=160):
        """
        Args:
            model_path: Path đến YOLO face detection model
            margin: Margin khi crop (pixels) - PHẢI GIỐNG DATASET!
            image_size: Target size sau khi crop - PHẢI GIỐNG DATASET!
        """
        print(f"[INFO] Loading YOLO face detection model: {model_path}")
        self.model = YOLO(model_path)
        self.margin = margin
        self.image_size = image_size
        print(f"[OK] YOLO face detector loaded (margin={margin}, size={image_size})")
    
    def detect_and_align(self, image, detect_multiple_faces=False):
        """WRAPPED với try-except để catch errors"""
        try:
            return self._detect_and_align_internal(image, detect_multiple_faces)
        except Exception as e:
            print(f"[ERROR in YOLO detector] {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_and_align_internal(self, image, detect_multiple_faces=False):
        """
        Detect và align faces - GIỐNG HỆT logic tạo dataset
        
        Args:
            image: BGR image (numpy array hoặc PIL Image)
            detect_multiple_faces: Detect nhiều faces hay chỉ 1 (center, largest)
        
        Returns:
            List of aligned faces và bounding boxes
            [
                {
                    'aligned_face': PIL Image (160x160),
                    'bbox': (x1, y1, x2, y2) - original bbox,
                    'bbox_with_margin': (x1, y1, x2, y2) - bbox sau khi thêm margin
                },
                ...
            ]
        """
        # Convert to PIL Image nếu cần
        if isinstance(image, np.ndarray):
            # Assume BGR from OpenCV
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_np = image
        else:
            image_pil = image
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        img_size = np.asarray(img_np.shape)[0:2]  # (height, width)
        
        # Run YOLO detection
        results = self.model(image_pil, verbose=False)
        
        aligned_faces = []
        
        for result in results:
            bounding_boxes = result.boxes
            nrof_faces = len(bounding_boxes)
            
            if nrof_faces == 0:
                print("[DEBUG] No faces detected by YOLO")
                continue
            
            det_arr = []
            
            # LOGIC GIỐNG HỆT CODE DATASET PREPROCESSING
            if nrof_faces > 1:
                if detect_multiple_faces:
                    # Detect tất cả faces
                    for i in range(nrof_faces):
                        det_arr.append(bounding_boxes.xyxy[i].cpu().numpy())
                else:
                    # Chọn face TỐT NHẤT (center + size weighted)
                    # LOGIC CHÍNH XÁC từ code dataset
                    bounding_box_size = (bounding_boxes.xywh[:, 2] * bounding_boxes.xywh[:, 3])
                    img_center = img_size / 2
                    
                    # Calculate offsets from center
                    offsets = np.vstack([
                        (bounding_boxes.xyxy[:, 0] + bounding_boxes.xyxy[:, 2]) / 2 - img_center[1],
                        (bounding_boxes.xyxy[:, 1] + bounding_boxes.xyxy[:, 3]) / 2 - img_center[0]
                    ])
                    
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    
                    # Index: Largest face ở center nhất
                    index = np.argmax(bounding_box_size.cpu().numpy() - offset_dist_squared * 2.0)
                    det_arr.append(bounding_boxes.xyxy[index, :].cpu().numpy())
            else:
                # Chỉ 1 face
                det_arr.append(bounding_boxes.xyxy[0, :].cpu().numpy())
            
            # Process mỗi detected face
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                
                # LOGIC CROP CHÍNH XÁC từ code dataset
                bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                
                # Crop
                cropped = img_np[bb[1]:bb[3], bb[0]:bb[2], :]
                
                # Resize to target size (160x160 như dataset)
                scaled = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).resize(
                    (self.image_size, self.image_size), 
                    resample=Image.BILINEAR
                )
                
                aligned_faces.append({
                    'aligned_face': scaled,  # PIL Image 160x160
                    'aligned_face_np': np.array(scaled),  # Numpy array
                    'bbox': det.astype(int).tolist(),  # Original bbox [x1, y1, x2, y2]
                    'bbox_with_margin': bb.tolist()    # Bbox với margin
                })
        
        return aligned_faces
    
    def detect_faces_cv2(self, image_bgr, detect_multiple_faces=False):
        """
        Detect faces và return aligned faces as numpy arrays (BGR)
        
        Args:
            image_bgr: BGR image (OpenCV format)
            detect_multiple_faces: Detect multiple or single best face
        
        Returns:
            List of dicts với aligned faces (BGR numpy arrays)
        """
        aligned_results = self.detect_and_align(image_bgr, detect_multiple_faces)
        
        # Convert PIL to BGR numpy
        for result in aligned_results:
            pil_img = result['aligned_face']
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            result['aligned_face_bgr'] = bgr_img
        
        return aligned_results


# Global instance
yolo_face_detector = None


def get_yolo_face_detector(model_path='yolov8n-face.pt', margin=32, image_size=160):
    """
    Get global YOLO face detector instance (singleton pattern)
    
    Args:
        model_path: YOLO face model path
        margin: Margin khi crop (PHẢI GIỐNG DATASET!)
        image_size: Output size (PHẢI GIỐNG DATASET!)
    
    Returns:
        YOLOFaceDetector instance
    """
    global yolo_face_detector
    
    if yolo_face_detector is None:
        yolo_face_detector = YOLOFaceDetector(
            model_path=model_path,
            margin=margin,
            image_size=image_size
        )
    
    return yolo_face_detector


if __name__ == "__main__":
    # Test YOLO face detector
    print("Testing YOLO Face Detector...")
    print("="*60)
    
    detector = YOLOFaceDetector()
    
    # Test với ảnh mẫu (nếu có)
    test_img_path = "test_face.jpg"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        results = detector.detect_faces_cv2(img)
        
        print(f"\nDetected {len(results)} face(s)")
        for i, face_result in enumerate(results):
            print(f"Face {i+1}:")
            print(f"  - Original bbox: {face_result['bbox']}")
            print(f"  - Bbox with margin: {face_result['bbox_with_margin']}")
            print(f"  - Aligned face shape: {face_result['aligned_face_bgr'].shape}")
    else:
        print(f"Test image not found: {test_img_path}")
        print("Module ready to use!")
    
    print("="*60)

