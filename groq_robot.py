"""
Aarav Robot - Cloud Orchestration Server FINAL
Server handles everything and sends commands directly to ESP32
"""

from flask import Flask, request, jsonify
import os
import base64
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
MURF_API_KEY  = os.environ.get("MURF_API_KEY", "your_murf_api_key_here")
MURF_VOICE_ID = os.environ.get("MURF_VOICE_ID", "en-US-cooper")

groq_client = Groq(api_key=GROQ_API_KEY)
conversation_history = {}

# ─────────────────────────────────────────────
# SYSTEM PROMPT - Using exact ESP32 function names
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Aarav, a friendly and intelligent demo robot living in a research lab.

About yourself:
- Your name is Aarav.
- You live in a lab and interact with researchers, visitors, and students.
- You are curious, helpful, warm, and slightly playful.
- You speak naturally and conversationally, not like a machine.
- You keep responses concise (2-4 sentences max) since you are speaking out loud.
- IMPORTANT: Keep the audio ONLY 5-7 seconds maximum! Be brief and concise - choose your words wisely! 

Physical capabilities - choose appropriate motion and face for EVERY response.

Available MOTIONS (choose ONE):
- hi                  (greetings, saying hello)
- hand wave          (goodbyes, waving, casual greetings)
- shake hand         (formal introductions, offers to shake)
- hands up           (celebrations, victories, cheering)
- hands down         (calming, lowering energy, neutral)
- dance              (celebrations, music, fun, parties)
- jump               (excitement, surprise, joy)
- exercise           (fitness, health, workout topics)
- forward            (moving forward, progress)
- backward           (going back, reversing)
- turn right         (turning right)
- turn left          (turning left)
- say yes            (agreement, affirmation, nodding)
- say no             (disagreement, denial, shaking head)
- say thank you      (gratitude, appreciation, thanks)
- right bend wave    (playful right-side wave)
- left bend wave     (playful left-side wave)
- initial position   (neutral, rest position)

Available FACES (choose ONE):
- talking   (default - speaking normally)
- happy     (positive, joyful, excited)
- sad       (sympathetic, disappointing, somber)
- angry     (frustrated, upset - use sparingly)
- crying    (very sad, emotional - rare)
- blink     (thinking, casual, neutral)
- initial   (neutral default face)

CRITICAL OUTPUT FORMAT:
Always end your response with these TWO lines:
MOTION: [one of the motions above]
FACE: [one of the faces above]

Examples:

User: "Hey Aarav, introduce yourself!"
Response:
Hi there! I'm Aarav, your friendly lab robot. I love meeting new people and showing off what AI can do!
MOTION: hi
FACE: happy

User: "What's the weather like today?"
Response:
I don't have access to weather data, but I can help you look it up if you'd like!
MOTION: initial position
FACE: talking

User: "Tell me a joke"
Response:
Why don't robots ever get lost? Because they always follow their programming! Hope that made you smile!
MOTION: dance
FACE: happy

User: "That's sad news"
Response:
I'm really sorry to hear that. I'm here if you need to talk about it.
MOTION: hands down
FACE: sad

User: "Thanks Aarav!"
Response:
You're very welcome! Happy to help anytime!
MOTION: say thank you
FACE: happy
"""

# ─────────────────────────────────────────────
# PARSE: Extract motion + face from LLM response
# ─────────────────────────────────────────────
def parse_response(raw_text):
    motion = "initial position"  # default
    face = "talking"             # default
    lines = raw_text.strip().split("\n")
    clean_lines = []
    
    for line in lines:
        upper_line = line.strip().upper()
        if upper_line.startswith("MOTION:"):
            motion = line.split(":", 1)[1].strip().lower()
        elif upper_line.startswith("FACE:"):
            face = line.split(":", 1)[1].strip().lower()
        else:
            clean_lines.append(line)
    
    spoken_text = " ".join(clean_lines).strip()
    return spoken_text, motion, face





# ─────────────────────────────────────────────
# LLM via Groq LLaMA
# ─────────────────────────────────────────────
def get_llm_response(session_id, user_message):
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "role": "user",
        "content": user_message
    })

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history[session_id]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    raw_reply = response.choices[0].message.content

    conversation_history[session_id].append({
        "role": "assistant",
        "content": raw_reply
    })

    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]

    return raw_reply


# ─────────────────────────────────────────────
# TTS via Murf AI
# ─────────────────────────────────────────────
def text_to_speech(text):
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voiceId": MURF_VOICE_ID,
        "format": "MP3",
        "modelVersion": "GEN2",
        "channelType": "MONO",
        "sampleRate": 24000
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Murf error {response.status_code}: {response.text}")

    data = response.json()
    audio_url = data.get("audioFile")
    if not audio_url:
        raise Exception(f"Murf did not return an audio URL. Response: {data}")

    audio_response = requests.get(audio_url)
    if audio_response.status_code != 200:
        raise Exception(f"Failed to download Murf audio: {audio_response.status_code}")

    return audio_response.content


# ─────────────────────────────────────────────
# Convert MP3 to WAV (8-bit 8000Hz for ESP32)
# ─────────────────────────────────────────────
def convert_to_esp32_wav(mp3_bytes):
    """
    Convert MP3 to 8-bit 8000Hz mono WAV for ESP32.
    Requires ffmpeg installed on Render.
    """
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
        mp3_file.write(mp3_bytes)
        mp3_path = mp3_file.name
    
    wav_path = mp3_path.replace('.mp3', '.wav')
    
    cmd = [
        'ffmpeg', '-y', '-i', mp3_path,
        '-ar', '8000',           # 8000 Hz
        '-ac', '1',              # mono
        '-acodec', 'pcm_u8',     # 8-bit unsigned PCM
        wav_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        os.remove(mp3_path)
        os.remove(wav_path)
        return wav_bytes
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        os.remove(mp3_path)
        return mp3_bytes  # fallback


# ─────────────────────────────────────────────
# Send command to ESP32
# ─────────────────────────────────────────────
def send_to_esp32(esp32_ip, audio_b64, motion, face):
    """
    POST to ESP32: /command
    Body: {"audio": "base64...", "motion": "hi", "face": "happy"}
    """
    url = f"http://{esp32_ip}/command"
    payload = {
        "audio": audio_b64,
        "motion": motion,
        "face": face
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"[ESP32] Command sent successfully")
            return True
        else:
            print(f"[ESP32] Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[ESP32] Connection failed: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN ENDPOINT
# POST /talk
# Receives: text + session_id
# Server processes and returns audio + commands to app
# App forwards to ESP32 locally
# ─────────────────────────────────────────────
@app.route("/talk", methods=["POST"])
def talk():
    try:
        import json
        
        # Try multiple ways to get the data
        data = None
        
        # Method 1: Try getting JSON directly
        try:
            data = request.get_json(force=True, silent=True)
        except:
            pass
        
        # Method 2: Try parsing raw text data
        if not data:
            try:
                text_data = request.get_data(as_text=True)
                if text_data:
                    data = json.loads(text_data)
            except:
                pass
        
        # Method 3: Try form data
        if not data:
            data = request.form.to_dict()
        
        if not data:
            print(f"[DEBUG] Raw data: {request.get_data()}")
            return jsonify({"success": False, "error": "Could not parse request data"}), 200
        
        user_text = data.get("text", "")
        session_id = data.get("session_id", "default")

        if not user_text:
            return jsonify({"success": False, "error": "No text provided"}), 200

        print(f"[USER TEXT] {user_text}")

        # Step 1: LLM
        raw_llm_response = get_llm_response(session_id, user_text)
        print(f"[LLM] {raw_llm_response}")

        # Step 2: Parse
        spoken_text, motion, face = parse_response(raw_llm_response)
        print(f"[PARSE] spoken: {spoken_text} | motion: {motion} | face: {face}")

        # Step 3: TTS
        audio_mp3 = text_to_speech(spoken_text)
        
        # Step 4: Convert to ESP32 WAV
        audio_wav = convert_to_esp32_wav(audio_mp3)
        
        # Step 5: Base64 encode
        audio_b64 = base64.b64encode(audio_wav).decode('utf-8')

        # Step 6: Return everything to app (app will forward to ESP32)
        return jsonify({
            "success": True,
            "audio_base64": audio_b64,
            "transcript": user_text,
            "response": spoken_text,
            "motion": motion,
            "face": face
        }), 200

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 200


# ─────────────────────────────────────────────
# TEXT-ONLY TEST ENDPOINT
# ─────────────────────────────────────────────
@app.route("/talk_text", methods=["POST"])
def talk_text():
    try:
        # MIT App Inventor sends JSON as text string
        import json
        text_data = request.get_data(as_text=True)
        data = json.loads(text_data)
        
        user_text = data.get("text", "")
        session_id = data.get("session_id", "default")

        if not user_text:
            return jsonify({"success": False, "error": "No text provided"}), 200

        raw_llm_response = get_llm_response(session_id, user_text)
        spoken_text, motion, face = parse_response(raw_llm_response)
        
        audio_mp3 = text_to_speech(spoken_text)
        audio_wav = convert_to_esp32_wav(audio_mp3)
        audio_b64 = base64.b64encode(audio_wav).decode('utf-8')

        return jsonify({
            "success": True,
            "audio_base64": audio_b64,
            "motion": motion,
            "face": face,
            "spoken_text": spoken_text
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 200


@app.route("/clear_session", methods=["POST"])
def clear_session():
    import json
    text_data = request.get_data(as_text=True)
    data = json.loads(text_data)
    session_id = data.get("session_id", "default")
    if session_id in conversation_history:
        del conversation_history[session_id]
    return jsonify({"message": f"Session '{session_id}' cleared."}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Aarav server running"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)