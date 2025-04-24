# AI4Bharat ASR Application

This project leverages NVIDIA NeMo's Automatic Speech Recognition (ASR) capabilities to transcribe speech into text. It is containerized using Docker for ease of deployment.

## 🚀 Features

- Uses NeMo ASR models for transcription
- REST API endpoint to submit audio and receive transcriptions
- Compatible with multiple backends including CPU and GPU
- Containerized using Docker for portability

---

## 📦 Installation

You can run this application in two ways:

### Option 1: Running with Docker

#### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/AI4Bharat.git
cd AI4Bharat
```

#### Step 2: Build the Docker image

```bash
docker build -t ai4bharat-app .
```

#### Step 3: Run the Docker container

```bash
docker run -p 5000:5000 ai4bharat-app
```

> 🧠 **Note:** You may see a warning about `ffmpeg` or `avconv` if they are not installed. For audio processing with `pydub`, install ffmpeg inside the Dockerfile or system-wide.

To access the app:
- Open your browser and navigate to `http://localhost:5000`

---

### Option 2: Running Locally via Virtual Environment

#### Step 1: Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

#### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

If you see any `ModuleNotFoundError`, you may manually install missing packages:

```bash
pip install lhotse einops transformers sentencepiece webdataset editdistance
```

#### Step 3: Run the application

```bash
python app.py
```

---

## 📤 API Usage

Once the server is running, send a POST request to the `/transcribe` endpoint:

### Example:

```bash
curl -X POST http://localhost:5000/transcribe \
  -F audio=@sample.wav
```

The response will be a JSON object with the transcribed text.

---

## 🐳 Docker Hub

To push your image to Docker Hub:

```bash
# Tag your image
docker tag ai4bharat-app yourusername/ai4bharat-app:latest

# Push it
docker push yourusername/ai4bharat-app:latest
```

---

## 🛠 Troubleshooting

- **`libgomp.so.1` missing?**
  Add the following line to your Dockerfile:
  ```Dockerfile
  RUN apt-get update && apt-get install -y libgomp1
  ```

- **`ffmpeg` not found?**
  Install `ffmpeg` in Docker:
  ```Dockerfile
  RUN apt-get update && apt-get install -y ffmpeg
  ```

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [PyTorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)
