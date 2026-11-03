"""
app.py  -  Streamlit UI for the Smart Photo Album
Run:  streamlit run app.py
"""
import os
import streamlit as st
from album import SmartAlbum

st.set_page_config(page_title="Smart Photo Album", page_icon="📸", layout="wide")
UPLOAD_DIR = "photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@st.cache_resource
def get_album():
    return SmartAlbum()


album = get_album()

# ---------------- sidebar ----------------
st.sidebar.title("📸 Smart Photo Album")
st.sidebar.caption("Face detection (YuNet) + recognition (SFace) + clustering (DBSCAN)")

files = st.sidebar.file_uploader("Add photos", type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True)
folder = st.sidebar.text_input("...or scan a folder", value=UPLOAD_DIR)
eps = st.sidebar.slider("Grouping strictness (eps)", 0.30, 0.80, 0.55, 0.01,
                        help="Lower = stricter (more groups). Higher = looser (fewer groups).")

if st.sidebar.button("🔍 Scan & group faces", type="primary"):
    for f in files or []:
        with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())
    bar = st.sidebar.progress(0.0)
    n = album.scan_folder(folder, progress=lambda p, name: bar.progress(p, text=os.path.basename(name)))
    album.cluster(eps=eps)
    st.sidebar.success(f"Scanned {n} new photos · {len(album.faces)} faces total")

if st.sidebar.button("Re-group only"):
    album.cluster(eps=eps)

if st.sidebar.button("🗑️ Reset album"):
    album.reset()
    st.rerun()

st.sidebar.metric("Photos", len(album.photos))
st.sidebar.metric("Faces found", len(album.faces))

# ---------------- main tabs ----------------
tab_people, tab_search, tab_face, tab_all = st.tabs(["👥 People", "🔎 Search by name", "🤳 Search by face", "🖼️ All photos"])
people = album.people()

with tab_people:
    if not people:
        st.info("Upload photos in the sidebar and hit **Scan & group faces**.")
    for cid, faces in people.items():
        name = album.name_of(cid)
        with st.expander(f"**{name}** — {len(faces)} faces in {len(album.photos_of(cid))} photos", expanded=False):
            st.image([f["crop_path"] for f in faces[:12]], width=90)
            if cid != -1:
                c1, c2 = st.columns(2)
                new = c1.text_input("Rename", value=name, key=f"name_{cid}")
                if new != name and new.strip():
                    album.rename(cid, new)
                    st.rerun()
                others = {album.name_of(o): o for o in people if o not in (cid, -1)}
                tgt = c2.selectbox("Merge into…", ["—"] + list(others), key=f"merge_{cid}")
                if tgt != "—" and c2.button("Merge", key=f"mbtn_{cid}"):
                    album.merge(others[tgt], cid)
                    st.rerun()
            st.markdown("**Album:**")
            cols = st.columns(4)
            for i, p in enumerate(album.photos_of(cid)):
                cols[i % 4].image(album.annotated(p), use_container_width=True)

with tab_search:
    q = st.text_input("Show all photos of…", placeholder="e.g. Nandana")
    if q:
        res = album.search_name(q)
        if not res:
            st.warning("No person with that name. Rename a group in the People tab first.")
        for nm, photos in res.items():
            st.subheader(f"{nm} ({len(photos)} photos)")
            cols = st.columns(4)
            for i, p in enumerate(photos):
                cols[i % 4].image(album.annotated(p), use_container_width=True)

with tab_face:
    st.write("Upload a selfie, and I'll find every photo that person appears in.")
    qimg = st.file_uploader("Query face", type=["jpg", "jpeg", "png"], key="qface")
    thr = st.slider("Match threshold (cosine similarity)", 0.2, 0.7, 0.363, 0.01)
    if qimg:
        qp = os.path.join("album_data", "_query" + os.path.splitext(qimg.name)[1])
        with open(qp, "wb") as out:
            out.write(qimg.getbuffer())
        st.image(qp, width=150)
        matches = album.search_by_face(qp, thr)
        st.write(f"**{len(matches)} matching photos**")
        cols = st.columns(4)
        for i, (p, s) in enumerate(matches):
            cols[i % 4].image(album.annotated(p), caption=f"similarity {s:.2f}", use_container_width=True)

with tab_all:
    cols = st.columns(4)
    for i, p in enumerate(album.photos):
        cols[i % 4].image(album.annotated(p), caption=os.path.basename(p), use_container_width=True)
