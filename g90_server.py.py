import sys, os, threading, numpy as np, pyaudio, time
from flask import Flask, render_template
from flask_socketio import SocketIO
from datetime import datetime, timezone

# --- CONFIGURAZIONE PERCORSI ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from PyFT8.cycle_manager import Cycle_manager
    from PyFT8.sigspecs import FT8
    print("MOTORE PyFT8 CARICATO CON SUCCESSO")
except ImportError:
    print("ERRORE: Cartella PyFT8 non trovata nello script path!")

app = Flask(__name__)
# Usiamo async_mode='threading' per evitare conflitti con i calcoli FFT
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Parametri Audio
FS = 12000
CHUNK = 1024
connected_users = 0

@socketio.on('connect')
def handle_connect():
    global connected_users
    connected_users += 1
    socketio.emit('user_count', {'count': connected_users})
    print(f"📡 Browser connesso. Utenti totali: {connected_users}")

@socketio.on('disconnect')
def handle_disconnect():
    global connected_users
    connected_users = max(0, connected_users - 1)
    socketio.emit('user_count', {'count': connected_users})

# --- LOGICA DECODER ---
def on_decode_callback(c):
    full_msg = f"{getattr(c, 'call_a', '')} {getattr(c, 'call_b', '')} {getattr(c, 'grid_rpt', '')}"
    # Estrazione smart nominativo
    parts = full_msg.replace("CQ ", "").split()
    call = parts[0] if parts and any(char.isdigit() for char in parts[0]) else "???"
    
    data = {
        'time': datetime.now(timezone.utc).strftime("%H:%M:%S"),
        'snr': f"{getattr(c, 'snr', 0):.0f}",
        'freq': f"{getattr(c, 'fHz', 0):.0f}",
        'msg': full_msg,
        'call': call
    }
    socketio.emit('new_msg', data)

def run_audio_engine():
    # Keywords per identificare lo Xiegu G90
    kw = ["USB", "Audio", "Codec", "G90"] 
    try:
        Cycle_manager(FT8, on_decode_callback, onOccupancy=None, input_device_keywords=kw)
    except Exception as e:
        print(f"Errore Decoder: {e}")

# --- LOGICA WATERFALL (OTTIMIZZATA PER FLUIDITÀ) ---
def run_waterfall():
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=FS, input=True, frames_per_buffer=CHUNK)
        print("Waterfall in ascolto...")
        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(raw, dtype=np.int16)
            
            # FFT e normalizzazione
            fft = np.abs(np.fft.rfft(data * np.hanning(CHUNK)))[:512]
            fft_norm = (np.clip(fft / 65, 0, 255)).astype(int).tolist()
            
            # Invia dati al browser
            socketio.emit('wf_data', fft_norm)
            
            # --- LIMITATORE DI FRAME (20 FPS) ---
            # Evita il sovraccarico del browser e risolve la lentezza
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Errore Waterfall: {e}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Avvio thread separati
    threading.Thread(target=run_audio_engine, daemon=True).start()
    threading.Thread(target=run_waterfall, daemon=True).start()
    
    print("Server avviato!")
    print("Apri il browser su: http://localhost:5000")

    socketio.run(app, host='0.0.0.0', port=5000)
