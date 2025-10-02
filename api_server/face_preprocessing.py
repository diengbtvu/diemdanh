"""
Advanced Face Preprocessing Module
Cải thiện chất lượng preprocessing để predict tốt nhất
"""

import cv2
import numpy as np
import torch
from PIL import Image

# Normalization constants (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class FacePreprocessor:
    """
    Advanced face preprocessor với nhiều techniques
    để đảm bảo kết quả predict tốt nhất
    """
    
    def __init__(self, target_size=224, margin_ratio=0.2):
        """
        Args:
            target_size: Kích thước ảnh output (224x224)
            margin_ratio: Tỷ lệ margin khi crop (0.2 = 20%)
        """
        self.target_size = target_size
        self.margin_ratio = margin_ratio
    
    def add_margin(self, bbox, img_shape, margin_ratio=None):
        """
        Thêm margin xung quanh face bounding box
        Giúp bao gồm thêm context (tóc, cằm, tai...)
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            img_shape: (height, width) của ảnh gốc
            margin_ratio: Tỷ lệ margin (default: self.margin_ratio)
        
        Returns:
            (x1, y1, x2, y2) với margin
        """
        if margin_ratio is None:
            margin_ratio = self.margin_ratio
        
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        
        # Tính width và height của bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Thêm margin
        margin_w = int(bbox_w * margin_ratio)
        margin_h = int(bbox_h * margin_ratio)
        
        # Apply margin
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def make_square_bbox(self, bbox, img_shape):
        """
        Chuyển bbox thành hình vuông (giữ face ở center)
        Tránh distortion khi resize
        
        Args:
            bbox: (x1, y1, x2, y2)
            img_shape: (height, width)
        
        Returns:
            (x1, y1, x2, y2) hình vuông
        """
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Lấy cạnh lớn nhất
        max_side = max(bbox_w, bbox_h)
        
        # Tính center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Tạo bbox vuông
        x1 = center_x - max_side / 2
        y1 = center_y - max_side / 2
        x2 = center_x + max_side / 2
        y2 = center_y + max_side / 2
        
        # Clip về image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        return x1, y1, x2, y2
    
    def enhance_face_quality(self, face_img):
        """
        Cải thiện chất lượng ảnh khuôn mặt
        
        Args:
            face_img: BGR image
        
        Returns:
            Enhanced BGR image
        """
        # 1. Histogram Equalization (cân bằng sáng)
        # Convert to LAB color space
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge và convert về BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def check_face_quality(self, face_img, bbox_area):
        """
        Kiểm tra chất lượng face image
        
        Args:
            face_img: Face image
            bbox_area: Diện tích bounding box
        
        Returns:
            (is_good_quality: bool, quality_score: float, reasons: list)
        """
        reasons = []
        quality_score = 1.0
        
        # 1. Check size (face quá nhỏ = low quality)
        if bbox_area < 40 * 40:  # Nhỏ hơn 40x40 pixels
            reasons.append("Face too small")
            quality_score *= 0.5
        
        # 2. Check blur (độ mờ)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:  # Ảnh mờ
            reasons.append(f"Blurry image (blur score: {laplacian_var:.1f})")
            quality_score *= 0.7
        
        # 3. Check brightness
        avg_brightness = np.mean(gray)
        if avg_brightness < 40:  # Quá tối
            reasons.append("Too dark")
            quality_score *= 0.8
        elif avg_brightness > 220:  # Quá sáng
            reasons.append("Too bright")
            quality_score *= 0.8
        
        is_good = quality_score > 0.5
        
        return is_good, quality_score, reasons
    
    def align_face(self, face_img, landmarks=None):
        """
        Căn chỉnh khuôn mặt (face alignment)
        Xoay để 2 mắt nằm ngang
        
        Args:
            face_img: Face image
            landmarks: Face landmarks (optional, từ MTCNN)
        
        Returns:
            Aligned face image
        """
        if landmarks is None:
            # Không có landmarks, return original
            return face_img
        
        # Lấy vị trí 2 mắt
        try:
            left_eye = landmarks[0]  # Left eye
            right_eye = landmarks[1]  # Right eye
            
            # Tính góc xoay
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Tính center để xoay
            center = ((left_eye[0] + right_eye[0]) / 2, 
                     (left_eye[1] + right_eye[1]) / 2)
            
            # Tạo rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Xoay ảnh
            h, w = face_img.shape[:2]
            aligned = cv2.warpAffine(face_img, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return aligned
            
        except Exception as e:
            # Nếu lỗi, return original
            print(f"[DEBUG] Face alignment failed: {e}")
            return face_img
    
    def preprocess_face(self, face_img, enhance=True):
        """
        Preprocessing pipeline hoàn chỉnh
        
        Args:
            face_img: BGR face image (already cropped)
            enhance: Có apply enhancement không
        
        Returns:
            Tensor ready for model
        """
        # 1. Enhance quality (optional)
        if enhance:
            face_img = self.enhance_face_quality(face_img)
        
        # 2. Resize to target size
        face_resized = cv2.resize(face_img, (self.target_size, self.target_size),
                                 interpolation=cv2.INTER_CUBIC)
        
        # 3. Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 4. Convert to tensor
        face_tensor = torch.from_numpy(face_rgb).float()
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # 5. Normalize to [0, 1]
        face_tensor = face_tensor / 255.0
        
        # 6. Normalize với ImageNet stats (giống training)
        for i in range(3):
            face_tensor[i] = (face_tensor[i] - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
        
        # 7. Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)
        
        return face_tensor
    
    def process_detected_face(self, frame, bbox, landmarks=None, 
                             enhance=True, align=True, quality_check=True):
        """
        Pipeline HOÀN CHỈNH để process một detected face
        
        Args:
            frame: Full image (BGR)
            bbox: (x1, y1, x2, y2) bounding box
            landmarks: Face landmarks từ MTCNN
            enhance: Apply enhancement
            align: Apply face alignment
            quality_check: Check quality trước khi process
        
        Returns:
            {
                'tensor': Face tensor for model,
                'quality_score': Quality score (0-1),
                'is_good_quality': bool,
                'quality_reasons': list of issues,
                'bbox_processed': Final bbox used
            }
        """
        x1, y1, x2, y2 = [int(b) for b in bbox]
        
        # 1. Add margin to bbox
        x1_m, y1_m, x2_m, y2_m = self.add_margin(
            (x1, y1, x2, y2), frame.shape, self.margin_ratio
        )
        
        # 2. Make square bbox
        x1_s, y1_s, x2_s, y2_s = self.make_square_bbox(
            (x1_m, y1_m, x2_m, y2_m), frame.shape
        )
        
        # 3. Crop face
        face_img = frame[y1_s:y2_s, x1_s:x2_s]
        
        # Check if crop is valid
        if face_img.size == 0:
            return None
        
        # 4. Quality check
        bbox_area = (x2 - x1) * (y2 - y1)
        is_good, quality_score, reasons = self.check_face_quality(face_img, bbox_area)
        
        if quality_check and not is_good:
            print(f"[DEBUG] Low quality face: {reasons}")
            # Vẫn process nhưng gắn flag
        
        # 5. Face alignment (if landmarks available)
        if align and landmarks is not None:
            # Adjust landmarks coordinates to cropped image
            landmarks_adjusted = landmarks.copy()
            landmarks_adjusted[:, 0] -= x1_s
            landmarks_adjusted[:, 1] -= y1_s
            face_img = self.align_face(face_img, landmarks_adjusted)
        
        # 6. Preprocess to tensor
        face_tensor = self.preprocess_face(face_img, enhance=enhance)
        
        return {
            'tensor': face_tensor,
            'quality_score': quality_score,
            'is_good_quality': is_good,
            'quality_reasons': reasons,
            'bbox_processed': (x1_s, y1_s, x2_s, y2_s),
            'original_bbox': (x1, y1, x2, y2)
        }


# Global instance
default_preprocessor = FacePreprocessor(target_size=224, margin_ratio=0.2)


def preprocess_face_advanced(frame, bbox, landmarks=None, 
                             enhance=True, align=True, quality_check=True):
    """
    Convenience function để preprocess face
    
    Args:
        frame: Full image (BGR)
        bbox: (x1, y1, x2, y2)
        landmarks: Face landmarks (optional)
        enhance: Apply enhancement
        align: Apply alignment
        quality_check: Check quality
    
    Returns:
        Preprocessed result dict hoặc None nếu failed
    """
    return default_preprocessor.process_detected_face(
        frame, bbox, landmarks, enhance, align, quality_check
    )


if __name__ == "__main__":
    # Test preprocessing
    print("Face Preprocessing Module")
    print("="*50)
    print("Features:")
    print("  ✓ Margin addition (context)")
    print("  ✓ Square bbox (no distortion)")
    print("  ✓ Quality enhancement (CLAHE)")
    print("  ✓ Quality checking (size, blur, brightness)")
    print("  ✓ Face alignment (rotation)")
    print("  ✓ Proper normalization (ImageNet)")
    print("="*50)

