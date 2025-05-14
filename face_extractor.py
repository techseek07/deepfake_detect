import numpy as np
import imageio.v3 as iio
import cv2
from face_detection import build_detector

class FaceExtractor:
    def __init__(self):
        # Use DSFDDetector from face-detection library
        self.detector = build_detector(
            "DSFDDetector", confidence_threshold=0.5, nms_iou_threshold=0.3
        )

    def extract_faces(self, video_path):
        """Extract faces using pure Python detector"""
        faces = []
        try:
            for frame in iio.imiter(video_path):
                # Convert to RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces (returns list of [x1, y1, x2, y2, confidence])
                detections = self.detector.detect(rgb_frame)
                
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    face_img = rgb_frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        faces.append(cv2.resize(face_img, (224, 224)))
        except Exception as e:
            print(f"Face extraction error: {str(e)}")
        return faces

# Singleton instance
face_extractor = FaceExtractor()

def extract_faces_from_video(video_path):
    return face_extractor.extract_faces(video_path)
