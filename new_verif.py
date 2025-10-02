import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch
# OUTPUT_IMAGE = "best_match.png"


def load_and_align_embedding(app, image_path, fallback_allowed=False):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        if fallback_allowed:
            print(f"⚠️ No face found in {image_path}, using fallback (resized image)")
            resized = cv2.resize(img, (112, 112))
            input_blob = resized.astype(np.float32)[..., ::-1]  # BGR to RGB
            input_blob = np.transpose(input_blob, (2, 0, 1))  # HWC to CHW
            input_blob = np.expand_dims(input_blob, axis=0)
            input_blob = (input_blob - 127.5) / 127.5
            embedding = app.models["recognition"].forward(input_blob)[0]
            return resized, embedding
        print(f"❌ No face found in {image_path}")
        return None, None
    face = faces[0]
    return face.crop_face, face.embedding

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def run_verif(id_image_path: str, frames_dir: str, output_path: str, id_display_path: str = None):
    """
    Compare embeddings between an ID image and multiple frames, save side-by-side best match.

    Args:
        id_image_path: path to cropped ID face image
        frames_dir: directory containing selected frame images
        output_path: file path to save the combined best_match image

    Returns:
        dict with best_match filename, score, average_score, status, and all_scores
    """
    # Validate ID image exists
    id_path = Path(id_image_path)
    if not id_path.exists():
        return {'error': 'ID image not found'}

    # Initialize face analysis model    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Face verification running on {device.upper()}")

    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract embedding for ID
    _, id_embedding = load_and_align_embedding(app, str(id_path), fallback_allowed=True)
    if id_embedding is None:
        return {'error': 'Could not extract ID embedding'}

    # Iterate through frames and compute similarity
    scores = []
    best_match_file = None
    best_match_score = -1.0

    for frame_file in sorted(Path(frames_dir).glob('*.png')):
        _, frame_embedding = load_and_align_embedding(app, str(frame_file), fallback_allowed=True)
        if frame_embedding is None:
            continue
        score = float(cosine_similarity([id_embedding], [frame_embedding])[0][0])
        scores.append((frame_file.name, round(score, 4)))
        if score > best_match_score:
            best_match_score = score
            best_match_file = frame_file.name

    if not scores or best_match_file is None:
        return {'error': 'No valid matches found'}

    average_score = sum(s for _, s in scores) / len(scores)

    # Determine verification status
    if best_match_score >= 0.4 and average_score >= 0.36:
        status = '✅ Verified'
    elif (average_score >= 0.24) and (average_score < 0.36):
        status = '🟡 Needs Manual Review'
    else:
        status = '❌ Unverified'

    # Create and save combined image
    display_path = Path(id_display_path) if id_display_path else id_path
    if not display_path.exists():
        print(f"⚠️ Display image {display_path} not found, falling back to {id_path}")
        display_path = id_path

    img_id = cv2.imread(str(display_path)) 
    img_best = cv2.imread(str(Path(frames_dir) / best_match_file))

    h = 300
    
    def resize_to_height(img, height):
        return cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
    id_resized = resize_to_height(img_id, h)
    best_resized = resize_to_height(img_best, h)
    combined = np.hstack((id_resized, best_resized))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)

    return {
        'best_match': best_match_file,
        'score': round(best_match_score, 4),
        'average_score': round(average_score, 4),
        'status': status,
        'image': str(out_path),
        'all_scores': scores
    }
