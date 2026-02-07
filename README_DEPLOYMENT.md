# 🚀 Hugging Face Spaces Deployment Guide

## Quick Setup (5 Minutes)

### Prerequisites
✅ GitHub repository already pushed  
✅ Hugging Face account created

---

## Step-by-Step Deployment

### 1. Create Hugging Face Account
- Go to: https://huggingface.co
- Click **"Sign Up"** or use GitHub OAuth
- Verify your email

### 2. Create New Space
- Click **"New"** → **"Space"**
- Fill in details:
  - **Owner**: Your username
  - **Space name**: `pneumonia-detection`
  - **License**: `mit`
  - **Select SDK**: **Docker**
  - **Visibility**: `Public` (or Private)
- Click **"Create Space"**

### 3. Connect GitHub Repository
Option A: **GitHub Integration** (Recommended)
- In your Space, go to **"Settings"** → **"Repository"**
- Click **"Connect to GitHub"**
- Authorize Hugging Face
- Select repository: `AI_Pneumonia-Detection_using_DenseNet121`
- Enable auto-sync

Option B: **Manual Git Push**
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-detection
git push hf main
```

### 4. Wait for Build
- Hugging Face will automatically:
  - Detect the Dockerfile
  - Build the Docker image
  - Deploy the container
- Build time: ~5-10 minutes
- Watch progress in the **"Logs"** tab

### 5. Access Your App
- Once deployed, your app will be live at:
  - `https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-detection`
- Click **"App"** tab to use it

---

## Troubleshooting

### Build Fails: "Model file too large"
**Solution**: Hugging Face supports large files via Git LFS
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes model2result.keras
git commit -m "Track model with Git LFS"
git push
```

### Port Binding Error
- Ensure `app.py` uses port **7860** (default for HF Spaces)
- Check: `port = int(os.environ.get("PORT", 7860))`

### Dependencies Missing
- Verify `requirements.txt` includes all packages
- Rebuild space: Settings → Factory reboot

### Model Loading Timeout
- This shouldn't happen on HF Spaces (no strict timeout)
- If it does, check logs for memory issues

---

## Alternative: One-Click Deploy

If you prefer a simpler option, you can also:

1. **Use Gradio** (easier UI framework for HF Spaces)
2. **Use Streamlit** (supported by HF Spaces)

But your current Flask app with Dockerfile works perfectly!

---

## Free Tier Limits
✅ **Unlimited usage** on free tier  
✅ **2 CPU cores**  
✅ **16GB RAM** (sufficient for your model)  
✅ **50GB storage**  
✅ **Cold start**: ~10-15 seconds after inactivity  

---

## Custom Domain (Optional)
- Go to Space Settings → **"Domains"**
- Add your custom domain
- Follow DNS configuration instructions

---

## Monitoring
- View logs: Space → **"Logs"** tab
- Check usage: Space → **"Analytics"** tab
- Restart: Space → **"Settings"** → **"Factory reboot"**

---

## Your Repository is Ready! 🎉
All necessary files are committed:
- ✅ `app.py` (Flask app with port 7860)
- ✅ `Dockerfile` (Docker configuration)
- ✅ `.dockerignore` (excludes unnecessary files)
- ✅ `requirements.txt` (dependencies)
- ✅ `model2result.keras` (trained model)
- ✅ `templates/index.html` (frontend)

**Next**: Go to huggingface.co and create your Space!
