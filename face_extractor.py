import numpy as np
import imageio.v3 as iio
from retinaface import RetinaFace

def extract_faces_from_video(video_path, target_size=(224, 224)):
    """Extract faces from video using imageio and RetinaFace"""
    faces = []
    
    try:
        # Read video using imageio
        for frame in iio.imiter(video_path):
            # Convert frame to RGB format
            rgb_frame = np.ascontiguousarray(frame[..., :3])
            
            # Detect faces
            faces_data = RetinaFace.detect_faces(rgb_frame)
            
            if isinstance(faces_data, dict):
                for face_id, face_info in faces_data.items():
                    facial_area = face_info['facial_area']
                    x1, y1, x2, y2 = facial_area
                    
                    # Extract and resize face
                    face_img = rgb_frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                        
                    resized_face = cv2.resize(face_img, target_size)
                    faces.append(resized_face)
                    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        
    return faces
