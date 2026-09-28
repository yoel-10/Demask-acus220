import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d

pg.setConfigOptions(antialias=True)

# --- 1. CONFIGURACIÓN Y AUDIO BASE ---
FILE_1 = "data/raw/kick.wav"  
FILE_2 = "data/raw/Bajo.wav"  
CHUNK = 4096  

def load_audio_files(f1, f2):
    global data1, data2, fs, min_len
    try:
        d1, sample_rate = sf.read(f1, dtype='float32')
        d2, _ = sf.read(f2, dtype='float32')
        if d1.ndim > 1: d1 = d1[:, 0]
        if d2.ndim > 1: d2 = d2[:, 0]
        mlen = min(len(d1), len(d2))
        return d1[:mlen], d2[:mlen], sample_rate, mlen
    except Exception as e:
        return np.zeros(4096), np.zeros(4096), 44100, 4096

data1, data2, fs, min_len = load_audio_files(FILE_1, FILE_2)
current_frame = 0
total_time_sec = min_len / fs if fs else 0

raw_fft_1 = np.full(CHUNK // 2 + 1, -100.0)
raw_fft_2 = np.full(CHUNK // 2 + 1, -100.0)
freqs = rfftfreq(CHUNK, 1/fs)
freqs[0] = 1 
target_freqs = np.logspace(np.log10(20), np.log10(20000), 500) 
log_target_freqs = np.log10(target_freqs)
display_fft_1 = np.full(500, -100.0)
display_fft_2 = np.full(500, -100.0)

current_rms_db = -100.0
current_peak_db = -100.0
kick_gain_db = 0.0 
bass_gain_db = 0.0 

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

# --- 2. HILO DE AUDIO ---
def audio_callback(outdata, frames, time, status):
    global current_frame, raw_fft_1, raw_fft_2, current_rms_db, current_peak_db
    if current_frame + frames > min_len:
        outdata[:] = 0; raise sd.CallbackStop 
        
    chunk1 = data1[current_frame : current_frame + frames].copy()
    chunk2 = data2[current_frame : current_frame + frames].copy()

    if kick_gain_db <= -99.0: chunk1 *= 0.0
    else: chunk1 *= (10 ** (kick_gain_db / 20.0))

    if bass_gain_db <= -99.0: chunk2 *= 0.0
    else: chunk2 *= (10 ** (bass_gain_db / 20.0))
        
    mixed = (chunk1 + chunk2) * 0.5
    outdata[:, 0] = mixed; outdata[:, 1] = mixed
    
    current_peak_db = 20 * np.log10(np.max(np.abs(mixed)) + 1e-10)
    current_rms_db = 20 * np.log10(np.sqrt(np.mean(mixed**2)) + 1e-10)
    
    window = np.hanning(frames)
    raw_fft_1 = 20 * np.log10(np.clip((np.abs(rfft(chunk1 * window)) / (CHUNK / 2)) * 2.0, 1e-10, 1.0))
    raw_fft_2 = 20 * np.log10(np.clip((np.abs(rfft(chunk2 * window)) / (CHUNK / 2)) * 2.0, 1e-10, 1.0))
    current_frame += frames

stream = sd.OutputStream(samplerate=fs, channels=2, blocksize=CHUNK, callback=audio_callback)

# --- 3. INTERFAZ TIPO DEMO ---
app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet("""
    QWidget { background-color: #0d0f14; color: #9ca3af; font-family: 'Consolas', 'Courier New', monospace; }
    QPushButton { background: #1f2430; border: 1px solid #374151; border-radius: 3px; padding: 6px 14px; font-weight: bold; color: #e5e7eb; }
    QPushButton:hover { background: #374151; border: 1px solid #60a5fa; }
    QGroupBox { border: 1px solid #374151; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #60a5fa; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
    QLabel { background: transparent; color: #d1d5db; }
""")

main_window = QtWidgets.QWidget()
main_window.setWindowTitle("DeMask")
main_window.resize(1100, 750)
main_layout = QtWidgets.QVBoxLayout(main_window)
main_layout.setContentsMargins(15, 15, 15, 15)

# BARRA SUPERIOR CON SOLO EL NOMBRE
top_panel = QtWidgets.QHBoxLayout()
lbl_title = QtWidgets.QLabel("📈 DeMask")
lbl_title.setStyleSheet("color: #60a5fa; font-weight: bold; font-size: 16px;")
top_panel.addWidget(lbl_title)
top_panel.addStretch()

lbl_status = QtWidgets.QLabel("ESTADO: EN ESPERA")
lbl_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px; background: #1f2430; padding: 3px 8px; border-radius: 3px;")
top_panel.addWidget(lbl_status)
main_layout.addLayout(top_panel)

# GRÁFICO ESPECTRAL
analyzer_layout = QtWidgets.QHBoxLayout()
graph_frame = QtWidgets.QFrame()
graph_frame.setStyleSheet("background-color: #08090c; border: 1px solid #1f2430; border-radius: 4px;")
graph_vbox = QtWidgets.QVBoxLayout(graph_frame)
graph_vbox.setContentsMargins(0,0,0,0)

plot_widget = pg.PlotWidget()
plot_widget.setBackground('transparent') 
plot_widget.showGrid(x=True, y=True, alpha=0.15) 
plot_widget.setMouseEnabled(x=False, y=False) 
plot_widget.setXRange(np.log10(20), np.log10(20000), padding=0)
plot_widget.setYRange(-90, 0, padding=0) 

plot_widget.hideAxis('left'); plot_widget.showAxis('right') 
y_axis = plot_widget.getAxis('right'); y_axis.setPen('#374151'); y_axis.setTextPen('#6b7280')
y_axis.setTicks([[ (v, f"{v}") for v in [0, -10, -20, -30, -40, -50, -60, -70, -80, -90] ]])

bottom_axis = plot_widget.getAxis('bottom'); bottom_axis.setPen('#374151'); bottom_axis.setTextPen('#6b7280')
bottom_axis.setTicks([[(np.log10(hz), f"{hz}Hz" if hz < 1000 else f"{hz//1000}kHz") for hz in [50, 100, 200, 500, 1000, 2000, 5000, 10000]]])

curve1 = plot_widget.plot(pen=pg.mkPen('#06b6d4', width=1.5), fillLevel=-90, brush=pg.mkBrush(6, 182, 212, 20)) 
curve2 = plot_widget.plot(pen=pg.mkPen('#ec4899', width=1.5), fillLevel=-90, brush=pg.mkBrush(236, 72, 153, 20)) 

graph_vbox.addWidget(plot_widget)
analyzer_layout.addWidget(graph_frame, stretch=4)

# PANEL DE TELEMETRÍA LATERAL
telemetry_group = QtWidgets.QGroupBox("TELEMETRÍA DSP")
telemetry_layout = QtWidgets.QVBoxLayout(telemetry_group)
telemetry_layout.setSpacing(10)

lbl_sr = QtWidgets.QLabel(f"Sample Rate: {fs} Hz")
lbl_chunk = QtWidgets.QLabel(f"Buffer Size: {CHUNK} samples")
lbl_fft_res = QtWidgets.QLabel(f"Resolución FFT: {fs/CHUNK:.2f} Hz/bin")
lbl_rms_val = QtWidgets.QLabel("RMS Master: -100.0 dB")
lbl_peak_val = QtWidgets.QLabel("Peak Master: -100.0 dB")

for lbl in [lbl_sr, lbl_chunk, lbl_fft_res, lbl_rms_val, lbl_peak_val]:
    lbl.setStyleSheet("color: #9ca3af; font-size: 11px;")
    telemetry_layout.addWidget(lbl)

telemetry_layout.addStretch()
analyzer_layout.addWidget(telemetry_group, stretch=1)
main_layout.addLayout(analyzer_layout)

# CONTROLES INFERIORES
controls_group = QtWidgets.QGroupBox("BANCO DE PRUEBAS Y GANANCIAS")
controls_layout = QtWidgets.QHBoxLayout(controls_group)

btn_play = QtWidgets.QPushButton("▶ INICIAR STREAM AUDIO")
btn_pause = QtWidgets.QPushButton("⏸ PAUSAR")
btn_restart = QtWidgets.QPushButton("⏮ REINICIAR")

def start_stream():
    stream.start()
    lbl_status.setText("ESTADO: PROCESANDO STREAM")
    lbl_status.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px; background: #1f2430; padding: 3px 8px; border-radius: 3px;")

def pause_stream():
    stream.stop()
    lbl_status.setText("ESTADO: PAUSADO")
    lbl_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px; background: #1f2430; padding: 3px 8px; border-radius: 3px;")

def restart_stream():
    global current_frame
    current_frame = 0
    if not stream.active: stream.start()
    lbl_status.setText("ESTADO: REINICIADO")
    lbl_status.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px; background: #1f2430; padding: 3px 8px; border-radius: 3px;")

btn_play.clicked.connect(start_stream)
btn_pause.clicked.connect(pause_stream)
btn_restart.clicked.connect(restart_stream)

controls_layout.addWidget(btn_play)
controls_layout.addWidget(btn_pause)
controls_layout.addWidget(btn_restart)
controls_layout.addSpacing(30)

slider_kick_layout = QtWidgets.QVBoxLayout()
lbl_k_text = QtWidgets.QLabel("Ganancia Kick: 0.0 dB")
slider_kick = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_kick.setRange(0, 100); slider_kick.setValue(50)
def update_kick_slider(val):
    global kick_gain_db
    kick_gain_db = (val / 100.0) * 24.0 - 12.0 if val > 0 else -100.0
    lbl_k_text.setText(f"Ganancia Kick: {kick_gain_db:+.1f} dB")
slider_kick.valueChanged.connect(update_kick_slider)
slider_kick_layout.addWidget(lbl_k_text); slider_kick_layout.addWidget(slider_kick)
controls_layout.addLayout(slider_kick_layout)

slider_bass_layout = QtWidgets.QVBoxLayout()
lbl_b_text = QtWidgets.QLabel("Ganancia Bass: 0.0 dB")
slider_bass = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_bass.setRange(0, 100); slider_bass.setValue(50)
def update_bass_slider(val):
    global bass_gain_db
    bass_gain_db = (val / 100.0) * 24.0 - 12.0 if val > 0 else -100.0
    lbl_b_text.setText(f"Ganancia Bass: {bass_gain_db:+.1f} dB")
slider_bass.valueChanged.connect(update_bass_slider)
slider_bass_layout.addWidget(lbl_b_text); slider_bass_layout.addWidget(slider_bass)
controls_layout.addLayout(slider_bass_layout)

main_layout.addWidget(controls_group)

# --- 4. BUCLE DE ACTUALIZACIÓN ---
def update_gui():
    global display_fft_1, display_fft_2
    if stream.active:
        s1 = gaussian_filter1d(np.interp(target_freqs, freqs, raw_fft_1), sigma=1.5)
        s2 = gaussian_filter1d(np.interp(target_freqs, freqs, raw_fft_2), sigma=4.0) 
        display_fft_1 = np.where(s1 > display_fft_1, s1 * 0.85 + display_fft_1 * 0.15, s1 * 0.10 + display_fft_1 * 0.90)
        display_fft_2 = np.where(s2 > display_fft_2, s2 * 0.85 + display_fft_2 * 0.15, s2 * 0.10 + display_fft_2 * 0.90)
        curve1.setData(log_target_freqs, display_fft_1)
        curve2.setData(log_target_freqs, display_fft_2)
        
        lbl_rms_val.setText(f"RMS Master: {current_rms_db:.1f} dB")
        lbl_peak_val.setText(f"Peak Master: {current_peak_db:.1f} dB")

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(30) 

main_window.show()
sys.exit(app.exec())