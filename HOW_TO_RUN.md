# 🎭 Meme Bully Classifier - How to Run

**Quick Start Guide** | Just 3 Steps!

---

## Step 1️⃣: Open PowerShell

Click on Windows Start → Search **PowerShell** → Open

---

## Step 2️⃣: Copy & Paste This

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
cd C:\Users\aryan\Desktop\MemeClassifier_Project
python -m venv meme_env
meme_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit Pillow numpy opencv-python
streamlit run app.py
```

---

## Step 3️⃣: Use the App!

✅ Browser opens automatically  
✅ Upload a meme image  
✅ Get classification (Bully / Non-Bully)  
✅ Rate the meme  
✅ Click "Save Results"  

**Done!** 🎉

---

## What It Does

| Feature | What Happens |
|---------|--------------|
| 📤 Upload | Choose any meme image (JPG, PNG, GIF) |
| 🎯 Classify | Auto-classifies as BULLY or NON-BULLY |
| ⭐ Rate | Rate Humor, Offensiveness, Relevance |
| 💾 Save | Results saved as JSON file |

---

## Results Location

Results saved in: `results/` folder as JSON files

```
MemeClassifier_Project/
├── results/
│   ├── meme_20260420_100000.json
│   ├── meme_20260420_101500.json
│   └── ...
```

---

## Stop the App

Press: **Ctrl + C** in PowerShell

---

## Run Again Next Time

```powershell
cd C:\Users\aryan\Desktop\MemeClassifier_Project
meme_env\Scripts\Activate.ps1
streamlit run app.py
```

---

## Troubleshooting

**Problem:** "Python not found"  
**Fix:** Install Python from python.org

**Problem:** "Streamlit not found"  
**Fix:** Run `pip install streamlit` again

**Problem:** "Permission denied"  
**Fix:** Run PowerShell as Administrator

---

**Questions?** Check README.md in the folder

**Enjoy! 🚀**
