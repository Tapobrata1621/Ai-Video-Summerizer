import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
import whisper

from summarizer import summarize_text

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "extracted_audio"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Faster model for CPU
model = whisper.load_model("tiny")

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    if video.audio is None:
        raise ValueError("This video has no audio track.")
    video.audio.write_audiofile(audio_path)
    video.close()

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    summary = ""
    error = ""
    uploaded_filename = ""

    if request.method == "POST":
        if "video" not in request.files:
            error = "No file part found."
            return render_template(
                "index.html",
                transcript=transcript,
                summary=summary,
                error=error,
                uploaded_filename=uploaded_filename
            )

        file = request.files["video"]

        if file.filename == "":
            error = "Please choose a video file."
            return render_template(
                "index.html",
                transcript=transcript,
                summary=summary,
                error=error,
                uploaded_filename=uploaded_filename
            )

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                uploaded_filename = filename

                video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(video_path)

                audio_filename = os.path.splitext(filename)[0] + ".mp3"
                audio_path = os.path.join(app.config["AUDIO_FOLDER"], audio_filename)

                # Extract audio
                extract_audio(video_path, audio_path)

                # Transcribe audio
                print("Transcribing started...")
                result = model.transcribe(audio_path)
                print("Transcribing done")

                transcript = result["text"].strip()

                # Summarize transcript
                if transcript:
                    summary = summarize_text(transcript)
                else:
                    summary = "No speech detected in the video."

            except Exception as e:
                error = f"Error: {str(e)}"
        else:
            error = "Only mp4, mov, avi, and mkv files are allowed."

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary,
        error=error,
        uploaded_filename=uploaded_filename
    )

if __name__ == "__main__":
    app.run(debug=True)