# Smart Photo Album Using Face Detection and Recognition

Drop in a bunch of photos. The app finds every face, works out who's who, and makes a separate album per person, like Google Photos' "People" tab.

## Pipeline

```
Photo ──► [1] Detect faces (YuNet) ──► [2] Align + Embed (SFace, 128-d) ──► [3] Cluster (DBSCAN, cosine) ──► People albums
                                                                    └──► [4] Search: by name / by selfie
```

| Stage | Tool | What it does |
|---|---|---|
| Detection | YuNet CNN (OpenCV `FaceDetectorYN`) | Box + 5 landmarks (eyes, nose, mouth corners) per face |
| Alignment | `alignCrop` | Rotates/crops so eyes are level, 112×112 |
| Recognition | SFace CNN (OpenCV `FaceRecognizerSF`) | Face → 128-number embedding |
| Grouping | DBSCAN, cosine distance | Groups similar embeddings, no need to know number of people |
| Matching | Cosine similarity ≥ 0.363 | "Is this the same person?" |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Models are in `models/` (from OpenCV Model Zoo). If missing, download:
- https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

## Features
- Upload photos or scan a folder
- Auto-grouped "Person 1, Person 2…" albums, with group photos appearing in every member's album
- Rename a person, merge two groups that are the same person
- Search by name ("show all photos of X")
- Search by face (upload a selfie → every photo they're in)
- Boxes and names drawn on photos
- Album saved to `album_data/` (JSON + NumPy), so it persists between runs

## Evaluation (for the report)
Make `dataset/<person_name>/*.jpg` (5–15 photos per person), then:
```bash
python evaluate.py dataset
```
Gives detection rate, verification accuracy/precision/recall/ROC-AUC, clustering ARI/NMI/purity, and `eval_similarity.png`.

## Files
- `face_engine.py`: detection + recognition
- `album.py`: indexing, clustering, search, storage
- `app.py`: Streamlit UI
- `evaluate.py`: metrics + plot
