"""
evaluate.py  -  Numbers for your report.

Put a labelled test set like:
    dataset/
        nandana/  img1.jpg img2.jpg ...
        friend1/  ...
        friend2/  ...
(one folder per person, ideally single-face photos, 5-15 photos each)

Run:  python evaluate.py dataset
Outputs detection rate, verification accuracy, clustering quality, and
saves a similarity-histogram plot (eval_similarity.png) for the report.
"""
import os
import sys
import itertools
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from face_engine import FaceEngine, SAME_PERSON_THRESHOLD
from album import IMG_EXT


def main(root, eps=0.55):
    eng = FaceEngine()
    embs, labels, total, detected = [], [], 0, 0
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        for f in sorted(os.listdir(pdir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            total += 1
            dets, _ = eng.process_image(os.path.join(pdir, f))
            if dets:
                detected += 1
                d = max(dets, key=lambda d: d["bbox"][2] * d["bbox"][3])
                embs.append(d["embedding"])
                labels.append(person)
    embs, labels = np.array(embs), np.array(labels)
    print(f"\n== Detection ==\nImages: {total}  Face found: {detected}  Rate: {detected / max(total, 1):.1%}")

    # verification: every pair -> same person or not?
    same, sims = [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        same.append(labels[i] == labels[j])
        sims.append(float(embs[i] @ embs[j]))
    same, sims = np.array(same), np.array(sims)
    pred = sims >= SAME_PERSON_THRESHOLD
    tp, fp = np.sum(pred & same), np.sum(pred & ~same)
    fn, tn = np.sum(~pred & same), np.sum(~pred & ~same)
    print(f"\n== Verification (threshold {SAME_PERSON_THRESHOLD}) ==")
    print(f"Pairs: {len(same)}  Accuracy: {(tp + tn) / len(same):.1%}")
    print(f"Precision: {tp / max(tp + fp, 1):.1%}  Recall: {tp / max(tp + fn, 1):.1%}")
    if same.any() and (~same).any():
        print(f"ROC-AUC: {roc_auc_score(same, sims):.3f}")

    # clustering
    pred_c = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(embs)
    print(f"\n== Clustering (DBSCAN eps={eps}) ==")
    print(f"True people: {len(set(labels))}  Groups found: {len(set(pred_c) - {-1})}")
    print(f"Adjusted Rand Index: {adjusted_rand_score(labels, pred_c):.3f}  (1.0 = perfect)")
    print(f"NMI: {normalized_mutual_info_score(labels, pred_c):.3f}")
    purity = sum(max(np.sum((pred_c == c) & (labels == l)) for l in set(labels)) for c in set(pred_c)) / len(labels)
    print(f"Purity: {purity:.1%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(sims[same], bins=30, alpha=0.7, label="Same person")
        plt.hist(sims[~same], bins=30, alpha=0.7, label="Different people")
        plt.axvline(SAME_PERSON_THRESHOLD, color="k", ls="--", label="Threshold")
        plt.xlabel("Cosine similarity"); plt.ylabel("Pairs"); plt.legend()
        plt.title("Face embedding similarity distribution")
        plt.tight_layout(); plt.savefig("eval_similarity.png", dpi=150)
        print("\nSaved eval_similarity.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
