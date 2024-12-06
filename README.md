# 🧠 AlumnAI
### Align Your Lecture Audio with PDF Slides, and Automatically Take Notes 📚🎙️

Welcome to **AlumnAI**, a tool that integrates **Speech-to-Text (STT)** with **PDF slide recognition** to create **automatic lecture notes**! 📝✨  
This repository is designed for students and professionals who want to save time while focusing on learning. Dive in and let AlumnAI handle the tedious work for you! 🚀

---

## 🌟 Features
- **High-quality Speech-to-Text (STT):** Fine-tuned for medical and technical terminology. 🩺⚙️
- **PDF Recognition System:** Extracts key information and aligns it with audio seamlessly. 📄↔️🎧
- **Automatic Note Generation:** Summarizes lectures into concise and meaningful notes. ✍️📋

---

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/alumnai.git
   cd alumnai
   ```

2. Create a virtual environment (Python 3.10.15):
   ```bash
   conda create -n alumnai python=3.10.15
   conda activate alumnai
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

Run the main script with the following command:
```bash
python main.py --pdf_path 'test/test.pdf' --audio_path 'test/chunk_010.mp3' --device 'cpu' or 'cuda' (--show) (--query)
```

### Arguments:
- **`--pdf_path`:** Path to the PDF file.
- **`--text_path`:** Path to the audio transcription file.
- **`--show`:** (Optional) Visualize the alignment process.
- **`--query`:** (Optional) Query style

---

## 🔍 Mechanism

AlumnAI combines **state-of-the-art technologies** to achieve its goal. Here's how:

### 1️⃣ High-Quality STT 🌐
- Fine-tuned **Faster-Whisper** to understand **medical terminology**.  
- **Dataset Creation:**  
  - Leveraging **Amazon Polly** to generate synthetic speech of English medical terms spoken with a Korean accent.  
  - Exploring other datasets like AI-hub’s ER conversations and OpenAI's STT outputs.  

- **Preprocessing Improvements:**  
  - Noise reduction techniques for better clarity.  

- **Postprocessing Enhancements:**  
  - **Error correction** using **LLMs** that analyze the entire context.  

---

### 2️⃣ PDF Recognition System 📄
- Extracts text from slides using **Llama 3.1**, a multimodal AI.  
- Embeds both STT and PDF text as vectors using **Amazon Titan v2**.  
- Aligns STT results with slides via **dynamic programming**, ensuring the best match.  

### 🚀 End Result
Once aligned, AlumnAI generates a clean, summarized version of your lecture notes! 🎉

---

## 🤝 Contributing

We welcome contributions to improve **AlumnAI**!  
Feel free to:
1. Open an issue for feature suggestions or bug reports. 🐛💡
2. Fork the repository and create a pull request. 🔀

Make sure to follow the [Contributor's Guide](CONTRIBUTING.md) for detailed instructions.  

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.

---

## 📞 Contact

Need help or have questions?  
Feel free to reach out:  
- **Email:** [alumnai-support@example.com](mailto:alumnai-support@example.com)  
- **Discord:** Join our community [here](https://discord.gg/yourlink).  

We’re always here to support you! ❤️  

---

**Unleash your productivity with AlumnAI! 🚀✨**
