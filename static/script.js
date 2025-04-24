let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordButton");
const stopBtn = document.getElementById("stopButton");
const transcriptEl = document.getElementById("transcript");

recordBtn.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    const lang = document.getElementById("lang").value;

    formData.append("audio", blob, "audio.wav");
    formData.append("lang", lang);

    transcriptEl.textContent = "Processing...";

    const response = await fetch("/transcribe", {
      method: "POST",
      body: formData
    });

    const result = await response.json();
    transcriptEl.textContent = result.transcription || "Error!";
  };

  mediaRecorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;
};

stopBtn.onclick = () => {
  mediaRecorder.stop();
  recordBtn.disabled = false;
  stopBtn.disabled = true;
};
