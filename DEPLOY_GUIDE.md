# Deploy MHA-GestureNet on Streamlit Community Cloud (Free)

## Prerequisites
- GitHub account (repo is already at https://github.com/Krishah27/MHA-GestureNet)
- Streamlit account — sign up free at https://share.streamlit.io

---

## Step 1 — Commit & Push All Changes

Open a terminal in your project folder and run:

```bash
# Stage all the deployment files
git add .gitignore requirements.txt packages.txt .streamlit/config.toml

# Stage the model weights (small files needed by the app)
git add -f weights/mha_gesturenet.h5 weights/label_encoder.pkl

# Commit
git commit -m "chore: prep for Streamlit Cloud deployment"

# Push
git push origin main
```

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** `Krishah27/MHA-GestureNet`
   - **Branch:** `main`
   - **Main file path:** `app/dashboard/streamlit_webrtc_app.py`
4. Click **"Deploy!"**

Streamlit will install all dependencies from `requirements.txt` + `packages.txt` automatically.  
Your app will be live at a URL like:  
`https://krishah27-mha-gesturenet-app-dashboard-streamlit-webrtc-app-xxxx.streamlit.app`

---

## Files Changed for Deployment

| File | What changed |
|------|-------------|
| `.gitignore` | Resolved merge conflict; allows `mha_gesturenet.h5` & `label_encoder.pkl` to be committed |
| `requirements.txt` | Pinned all versions for reproducible builds |
| `packages.txt` | System-level apt packages needed by OpenCV/MediaPipe |
| `.streamlit/config.toml` | Server config for cloud hosting |

---

## Notes

- **WebRTC** (camera access) works on Streamlit Cloud out of the box — users just need to allow camera permissions in browser.
- The app loads `weights/mha_gesturenet.h5` (1.1 MB) — small enough to commit directly to GitHub.
- `bilstm_model.keras` (24 MB) is **excluded** from git — it's not used by the app.
- Free tier limits: 1 GB RAM, 1 app always-on, sleeps after inactivity (wakes on visit).
