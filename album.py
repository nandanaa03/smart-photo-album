"""
album.py  -  Builds the smart album: scan photos -> cluster faces -> people albums.

Clustering: DBSCAN on cosine distance.
Analogy: every face is a dot on a map. Faces of the same person land close
together. DBSCAN draws circles of radius `eps` and groups dots that are
packed together, no need to tell it how many people there are.
"""
import os
import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from face_engine import FaceEngine, SAME_PERSON_THRESHOLD

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class SmartAlbum:
    def __init__(self, data_dir="album_data"):
        self.data_dir = data_dir
        self.crops_dir = os.path.join(data_dir, "faces")
        os.makedirs(self.crops_dir, exist_ok=True)
        self.engine = FaceEngine()
        self.faces = []      # [{id, photo, bbox, score, crop_path, cluster}]
        self.embeddings = np.zeros((0, 128), np.float32)
        self.names = {}      # cluster id (str) -> custom name
        self.photos = []     # every scanned photo path
        self.load()

    # ---------------- indexing ----------------
    def scan_folder(self, folder, progress=None):
        paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder)
                       for f in fs if f.lower().endswith(IMG_EXT))
        new = [p for p in paths if p not in self.photos]
        embs = list(self.embeddings)
        for i, p in enumerate(new):
            dets, _ = self.engine.process_image(p)
            self.photos.append(p)
            for d in dets:
                fid = len(self.faces)
                crop_path = os.path.join(self.crops_dir, f"face_{fid}.jpg")
                cv2.imwrite(crop_path, d["crop"])
                self.faces.append({"id": fid, "photo": p, "bbox": d["bbox"],
                                   "score": d["score"], "crop_path": crop_path,
                                   "cluster": -1})
                embs.append(d["embedding"])
            if progress:
                progress((i + 1) / len(new), p)
        self.embeddings = np.array(embs, np.float32).reshape(-1, 128)
        return len(new)

    # ---------------- grouping ----------------
    def cluster(self, eps=0.55, min_samples=1):
        """Group faces into people. Lower eps = stricter."""
        if len(self.faces) == 0:
            return {}
        old = {f["id"]: f["cluster"] for f in self.faces}
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(self.embeddings)
        # keep custom names if a renamed group still mostly maps to one new group
        new_names = {}
        for lab in set(labels) - {-1}:
            members = [f["id"] for f, l in zip(self.faces, labels) if l == lab]
            prev = [old[m] for m in members if str(old[m]) in self.names]
            if prev:
                top = max(set(prev), key=prev.count)
                new_names[str(lab)] = self.names[str(top)]
        for f, lab in zip(self.faces, labels):
            f["cluster"] = int(lab)
        self.names = new_names
        self.save()
        return self.people()

    def people(self):
        """cluster id -> list of faces, biggest group first."""
        groups = {}
        for f in self.faces:
            groups.setdefault(f["cluster"], []).append(f)
        return dict(sorted(groups.items(), key=lambda kv: (kv[0] == -1, -len(kv[1]))))

    def name_of(self, cid):
        if cid == -1:
            return "Unknown"
        return self.names.get(str(cid), f"Person {cid + 1}")

    def rename(self, cid, name):
        self.names[str(cid)] = name.strip()
        self.save()

    def merge(self, cid_keep, cid_drop):
        for f in self.faces:
            if f["cluster"] == cid_drop:
                f["cluster"] = cid_keep
        self.names.pop(str(cid_drop), None)
        self.save()

    def photos_of(self, cid):
        return sorted({f["photo"] for f in self.faces if f["cluster"] == cid})

    # ---------------- search ----------------
    def search_name(self, query):
        q = query.lower().strip()
        cids = [c for c in self.people() if q in self.name_of(c).lower()]
        return {self.name_of(c): self.photos_of(c) for c in cids}

    def search_by_face(self, image_path, threshold=SAME_PERSON_THRESHOLD):
        """Query with a selfie: return photos containing that person, best match first."""
        dets, _ = self.engine.process_image(image_path)
        if not dets or len(self.faces) == 0:
            return []
        q = max(dets, key=lambda d: d["bbox"][2] * d["bbox"][3])["embedding"]  # largest face
        sims = self.embeddings @ q
        best = {}
        for f, s in zip(self.faces, sims):
            if s >= threshold:
                best[f["photo"]] = max(best.get(f["photo"], -1), float(s))
        return sorted(best.items(), key=lambda kv: -kv[1])

    # ---------------- persistence ----------------
    def save(self):
        with open(os.path.join(self.data_dir, "album.json"), "w") as fp:
            json.dump({"faces": self.faces, "names": self.names, "photos": self.photos}, fp, indent=1)
        np.save(os.path.join(self.data_dir, "embeddings.npy"), self.embeddings)

    def load(self):
        j = os.path.join(self.data_dir, "album.json")
        e = os.path.join(self.data_dir, "embeddings.npy")
        if os.path.exists(j) and os.path.exists(e):
            with open(j) as fp:
                d = json.load(fp)
            self.faces, self.names, self.photos = d["faces"], d["names"], d.get("photos", [])
            self.embeddings = np.load(e)

    def reset(self):
        import shutil
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.__init__(self.data_dir)

    def annotated(self, photo):
        """Photo with boxes + names drawn on it."""
        img = self.engine._resize(cv2.imread(photo))
        for f in self.faces:
            if f["photo"] != photo:
                continue
            x, y, w, h = f["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 200, 120), 2)
            cv2.putText(img, self.name_of(f["cluster"]), (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
