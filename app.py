from flask import Flask, request, jsonify, render_template
import os
import torch
import tempfile
import nemo.collections.asr as nemo_asr
from utils import convert_to_wav, load_audio, preprocess_audio

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ai4bharat/IndicConformer"
model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name).to(device)
model.eval()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    lang_code = request.form.get('lang', 'hi')

    # Save and convert uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_input, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output:

        input_path = temp_input.name
        output_path = temp_output.name
        audio_file.save(input_path)

        # Convert to 16kHz wav
        convert_to_wav(input_path, output_path)

        # Load and preprocess
        waveform = load_audio(output_path)
        waveform = preprocess_audio(waveform).to(device)

        try:
            with torch.no_grad():
                result = model.transcribe([waveform.squeeze(0).cpu().numpy()], language_id=lang_code)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    return jsonify({"transcription": result[0]})

if __name__ == '__main__':
    app.run(debug=True)
