"""
face_engine.py  -  Face DETECTION + RECOGNITION (the "eyes" and the "brain")

Detection  : YuNet  (lightweight CNN face detector, ships with OpenCV)
Recognition: SFace  (CNN that turns a face into a 128-number "fingerprint" / embedding)

Analogy: YuNet draws a box around every face in a photo.
         SFace looks at each boxed face and writes down a 128-number
         "barcode". Two photos of the same person -> very similar barcodes.
"""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DET_MODEL = os.path.join(BASE, "models", "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(BASE, "models", "face_recognition_sface_2021dec.onnx")

# SFace's recommended cosine-similarity threshold for "same person"
SAME_PERSON_THRESHOLD = 0.363
MAX_SIDE = 1280  # downscale huge photos so detection stays fast


class FaceEngine:
    def __init__(self, det_conf=0.8):
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), det_conf, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")

    @staticmethod
    def _resize(img):
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
        return img

    def detect(self, img):
        """Return list of raw YuNet face rows (bbox, 5 landmarks, score)."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return [] if faces is None else list(faces)

    def embed(self, img, face):
        """Align the face using its landmarks, then get a unit-length embedding."""
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9), aligned

    def process_image(self, path):
        """Detect + embed every face in one photo."""
        img = cv2.imread(path)
        if img is None:
            return [], None
        img = self._resize(img)
        out = []
        for face in self.detect(img):
            emb, crop = self.embed(img, face)
            x, y, w, h = face[:4].astype(int)
            out.append({"bbox": [int(x), int(y), int(w), int(h)],
                        "score": float(face[-1]),
                        "embedding": emb,
                        "crop": crop})
        return out, img

    @staticmethod
    def similarity(e1, e2):
        return float(np.dot(e1, e2))  # cosine similarity (vectors are unit length)
