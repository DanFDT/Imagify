# 🎵 Imagify: Image-to-Song Recommender

Upload an image and get song recommendations that match its mood! 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎬 Demo 

### Analyze an Image and Get Song Recommendations
![Demo](demo/demo_video_1.gif)

### Different Moods, Different Songs
![Variety](demo/demo_video_2.gif)

---

## 🚀 How to Install

### 0. Install Python (Required)

If you don’t have Python installed, download it here:

🔗 https://www.python.org/downloads/

Make sure to check “Add Python to PATH” during installation.

Verify installation:
```bash
python --version 
pip --version
```
If both commands show a version number, you're ready to continue.
### 1. Clone the Repository
```bash
git clone https://github.com/DanFDT/Imagify.git
```
```bash
cd Imagify
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
```
```bash
venv\scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` and you can open the link in a browser

**Note:** First run takes 1-2 minutes to download the AI model (~600MB).

---

## 💡 How It Works

1. **Upload an image or choose from the samples** (JPG, JPEG, PNG)
2. **AI analyzes the mood** using OpenAI's CLIP model
3. **Get 10 song recommendations** from 1500+ most-streamed Spotify songs
4. **Click to open in Spotify or YouTube**

The AI detects 8 different moods:
- 😄 Happy & Energetic
- 🤩 Excited & Intense
- 😌 Calm & Peaceful
- 😢 Sad & Melancholic
- 🌙 Dark & Mysterious
- 💕 Romantic & Gentle
- 😠 Angry & Aggressive
- 🍂 Nostalgic & Reflective

---

## 🔧 Customize

### Change Match Quality

Edit `src/song_recommender.py` line 88:
```python
max_distance = 0.25  # Lower = stricter matching (0.25 for 85%+ matches)
```
---

## 🛠️ Built With

- **[CLIP](https://openai.com/research/clip)** - OpenAI's AI model for image understanding
- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[PyTorch](https://pytorch.org/)** - Deep learning
- **[Pandas](https://pandas.pydata.org/)** - Data processing

---

## 🙏 Credits

- **OpenAI** - [CLIP model](https://openai.com/research/clip)
- **Hugging Face** - [Transformers library](https://huggingface.co/transformers/)
- **Streamlit** - [Web framework](https://streamlit.io/)
- **Kaggle** - [Spotify dataset](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs?resource=download)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**DanFDT**  
GitHub: [@DANFDT](https://github.com/DanFDT)

---

