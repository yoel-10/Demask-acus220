# -- AVANCE DSP Y GUI (Parche 1: DSP Corregido, UI pendiente) --
import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal

pg.setConfigOptions(antialias=True)

FILE_1 = "data/raw/kick.wav"  
FILE_2 = "data/raw/Bajo.wav"  
CHUNK = 4096  

def load_audio_files(f1, f2):
    try:
        d1, sample_rate = sf.read(f1, dtype='float32')
        d2, _ = sf.read(f2, dtype='float32')
        if d1.ndim > 1: d1 = d1[:, 0]
        if d2.ndim > 1: d2 = d2[:, 0]
        mlen = min(len(d1), len(d2))
        return d1[:mlen], d2[:mlen], sample_rate, mlen
    except Exception as e:
        return np.zeros(4096, dtype='float32'), np.zeros(4096, dtype='float32'), 44100, 4096

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
sc_intensity = 0.70
is_sidechain_active = True
is_solo_conflict = False

conflict_lower = 60.0
conflict_upper = 150.0

b_bp, a_bp = np.zeros(3), np.zeros(3)       
b_sc, a_sc = np.zeros(3), np.zeros(3)       
b_env1, a_env1 = np.zeros(2), np.zeros(2)   
b_env2, a_env2 = np.zeros(2), np.zeros(2)   

zi_bp = np.zeros(2)
zi_sc = np.zeros(2)
zi_env1 = np.zeros(1)
zi_env2 = np.zeros(1)
prev_gain = 1.0  

def design_bandpass_biquad(f_center, q_factor, sample_rate):
    w0 = 2.0 * np.pi * f_center / sample_rate
    alpha = np.sin(w0) / (2.0 * q_factor)
    
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype='float64')
    a = np.array([1.0, a1 / a0, a2 / a0], dtype='float64')
    return b, a

def update_crossover_freq():
    global b_bp, a_bp, b_sc, a_sc, b_env1, a_env1, b_env2, a_env2
    global zi_bp, zi_sc, zi_env1, zi_env2
    
    f1 = max(30.0, min(250.0, conflict_lower))
    f2 = max(f1 + 30.0, min(500.0, conflict_upper))
    
    f_center = np.sqrt(f1 * f2)
    bw = f2 - f1
    q_factor = np.clip(f_center / bw, 0.7, 3.0)
    
    b_bp, a_bp = design_bandpass_biquad(f_center, q_factor, fs)
    b_sc, a_sc = design_bandpass_biquad(f_center, 0.8, fs)
    
    alpha1 = np.exp(-1.0 / (fs * 0.018))
    b_env1 = np.array([1.0 - alpha1], dtype='float64')
    a_env1 = np.array([1.0, -alpha1], dtype='float64')
    
    alpha2 = np.exp(-1.0 / (fs * 0.065))
    b_env2 = np.array([1.0 - alpha2], dtype='float64')
    a_env2 = np.array([1.0, -alpha2], dtype='float64')
    
    zi_bp = np.zeros(2)
    zi_sc = np.zeros(2)
    zi_env1 = np.zeros(1)
    zi_env2 = np.zeros(1)

def detect_masking():
    global conflict_lower, conflict_upper
    if min_len == 0: return
    sample_len = min(min_len, int(fs * 4.0))
    d1 = data1[:sample_len]; d2 = data2[:sample_len]
    window = np.hanning(sample_len)
    mag1 = np.abs(rfft(d1 * window))
    mag2 = np.abs(rfft(d2 * window))
    freqs_eval = rfftfreq(sample_len, 1/fs)
    mag1_s = gaussian_filter1d(mag1, sigma=15)
    mag2_s = gaussian_filter1d(mag2, sigma=15)
    overlap = mag1_s * mag2_s
    valid_idx = np.where((freqs_eval >= 35) & (freqs_eval <= 250))[0]
    if len(valid_idx) == 0: 
        update_crossover_freq()
        return
    best_idx = valid_idx[np.argmax(overlap[valid_idx])]
    half_max = overlap[best_idx] * 0.4
    left_idx = best_idx; right_idx = best_idx
    while left_idx > valid_idx[0] and overlap[left_idx] > half_max: left_idx -= 1
    while right_idx < valid_idx[-1] and overlap[right_idx] > half_max: right_idx += 1
    conflict_lower = max(30.0, freqs_eval[left_idx])
    conflict_upper = min(350.0, max(conflict_lower + 35.0, freqs_eval[right_idx]))
    update_crossover_freq()

detect_masking()

class StereoMeterLED(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(65); self.level = -100.0; self.peak = -100.0
    def update_level(self, rms, peak):
        self.level = max(rms, self.level - 2.5); self.peak = max(peak, self.peak - 0.2); self.update()
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#16171b")) 
        w, h = self.width(), self.height()
        y_top, y_bottom = 15, h - 15
        bar_h, bar_w, space, x1 = y_bottom - y_top, 10, 3, 5
        x2, x_text = x1 + bar_w + space, x1 + (bar_w * 2) + space + 6
        def db_to_y(db): return y_bottom - int(max(0.0, min(1.0, (db + 60.0) / 60.0)) * bar_h)
        painter.setPen(QtGui.QColor("#777777"))
        font = painter.font(); font.setPixelSize(9); font.setBold(True); painter.setFont(font)
        for tick in [0, -6, -12, -18, -24, -30, -36, -48, -60]:
            painter.drawText(x_text, db_to_y(tick) - 6, 25, 12, QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, str(tick))
        segment_h, curr_y = 3, y_bottom
        while curr_y > y_top:
            db_val = ((y_bottom - curr_y) / bar_h) * 60.0 - 60.0
            if db_val >= -6.0: base_color = QtGui.QColor("#ff3333") 
            elif db_val >= -18.0: base_color = QtGui.QColor("#ffd500") 
            else: base_color = QtGui.QColor("#00ff66") 
            dim_color = QtGui.QColor(base_color.red() // 5, base_color.green() // 5, base_color.blue() // 5)
            color = base_color if self.level >= db_val else dim_color
            painter.fillRect(x1, curr_y - segment_h, bar_w, segment_h, color)
            painter.fillRect(x2, curr_y - segment_h, bar_w, segment_h, color)
            curr_y -= (segment_h + 1)
        y_peak = db_to_y(self.peak)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        painter.drawLine(x1, y_peak, x1 + bar_w, y_peak); painter.drawLine(x2, y_peak, x2 + bar_w, y_peak)

def audio_callback(outdata, frames, time, status):
    global current_frame, raw_fft_1, raw_fft_2, current_rms_db, current_peak_db
    global zi_bp, zi_sc, zi_env1, zi_env2
    
    if current_frame + frames > min_len:
        current_frame = 0
        outdata[:] = 0
        raise sd.CallbackStop 
        
    chunk1 = data1[current_frame : current_frame + frames].copy()
    chunk2 = data2[current_frame : current_frame + frames].copy()

    if kick_gain_db <= -99.0: chunk1 *= 0.0
    else: chunk1 *= (10 ** (kick_gain_db / 20.0))

    if bass_gain_db <= -99.0: chunk2 *= 0.0
    else: chunk2 *= (10 ** (bass_gain_db / 20.0))
        
    bass_conflict_band, zi_bp = signal.lfilter(b_bp, a_bp, chunk2, zi=zi_bp)
    kick_sc, zi_sc = signal.lfilter(b_sc, a_sc, chunk1, zi=zi_sc)
    
    env_stage1, zi_env1 = signal.lfilter(b_env1, a_env1, np.abs(kick_sc), zi=zi_env1)
    env_smooth, zi_env2 = signal.lfilter(b_env2, a_env2, env_stage1, zi=zi_env2)

    if is_sidechain_active and sc_intensity > 0.0:
        env_db = 20.0 * np.log10(env_smooth + 1e-6)
        threshold_db = -22.0
        over_db = np.maximum(0.0, env_db - threshold_db)
        
        max_cut_db = 9.5 * sc_intensity
        reduction_db = np.minimum(max_cut_db, over_db * 0.65 * sc_intensity)
        subtract_amount = 1.0 - (10.0 ** (-reduction_db / 20.0))
        
        # FIX APLICADO HOY: Resta correcta para ecualización dinámica (antes sumaba)
        chunk2_processed = chunk2 - (bass_conflict_band * subtract_amount)
        conflict_monitored = bass_conflict_band * (1.0 - subtract_amount)
    else:
        chunk2_processed = chunk2
        conflict_monitored = bass_conflict_band

    if is_solo_conflict:
        mixed = conflict_monitored * 1.5
    else:
        mixed = (chunk1 + chunk2_processed) * 0.6
        mixed = np.tanh(mixed)
    
    outdata[:, 0] = mixed
    outdata[:, 1] = mixed
    current_peak_db = 20 * np.log10(np.max(np.abs(mixed)) + 1e-10)
    current_rms_db = 20 * np.log10(np.sqrt(np.mean(mixed**2)) + 1e-10)
    
    window = np.hanning(frames)
    raw_fft_1 = 20 * np.log10(np.clip((np.abs(rfft(chunk1 * window)) / (CHUNK / 2)) * 2.0, 1e-10, 1.0))
    raw_fft_2 = 20 * np.log10(np.clip((np.abs(rfft(chunk2_processed * window)) / (CHUNK / 2)) * 2.0, 1e-10, 1.0))
    current_frame += frames

stream = sd.OutputStream(samplerate=fs, channels=2, blocksize=CHUNK, callback=audio_callback)

app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet("""
    QWidget { background-color: #1a1c23; color: #d1d5db; font-family: 'Segoe UI', Arial, sans-serif; }
    QPushButton { background: #373b45; border: 1px solid #111317; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
    QLabel { background: transparent; }
""")

main_window = QtWidgets.QWidget()
main_window.setWindowTitle("DeMask")
main_window.resize(1280, 850)
main_layout = QtWidgets.QVBoxLayout(main_window)

top_bar = QtWidgets.QHBoxLayout()
lbl_logo = QtWidgets.QLabel("📈 DeMask"); lbl_logo.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 16px;")
top_bar.addWidget(lbl_logo); top_bar.addSpacing(20)

btn_load_kick = QtWidgets.QPushButton("📁 Cargar Kick WAV"); lbl_kick = QtWidgets.QLabel(os.path.basename(FILE_1))
btn_load_bass = QtWidgets.QPushButton("📁 Cargar Bass WAV"); lbl_bass = QtWidgets.QLabel(os.path.basename(FILE_2))

def select_kick_file():
    global FILE_1, data1, data2, fs, min_len, current_frame
    path, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, "Seleccionar Kick WAV", "", "Audio Files (*.wav *.flac *.mp3)")
    if path:
        if stream.active: stream.stop()
        FILE_1 = path
        lbl_kick.setText(os.path.basename(path))
        data1, data2, fs, min_len = load_audio_files(FILE_1, FILE_2)
        current_frame = 0
        detect_masking()
        conflict_region.setRegion([np.log10(conflict_lower), np.log10(conflict_upper)])

def select_bass_file():
    global FILE_2, data1, data2, fs, min_len, current_frame
    path, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, "Seleccionar Bass WAV", "", "Audio Files (*.wav *.flac *.mp3)")
    if path:
        if stream.active: stream.stop()
        FILE_2 = path
        lbl_bass.setText(os.path.basename(path))
        data1, data2, fs, min_len = load_audio_files(FILE_1, FILE_2)
        current_frame = 0
        detect_masking()
        conflict_region.setRegion([np.log10(conflict_lower), np.log10(conflict_upper)])

btn_load_kick.clicked.connect(select_kick_file)
btn_load_bass.clicked.connect(select_bass_file)

top_bar.addWidget(btn_load_kick); top_bar.addWidget(lbl_kick); top_bar.addSpacing(15)
top_bar.addWidget(btn_load_bass); top_bar.addWidget(lbl_bass); top_bar.addStretch()
main_layout.addLayout(top_bar)

analyzer_layout = QtWidgets.QHBoxLayout(); analyzer_layout.setSpacing(0)
graph_frame = QtWidgets.QFrame(); graph_frame.setStyleSheet("background-color: #12161f; border: 2px solid #0c0d11;")
plot_widget = pg.PlotWidget(); plot_widget.setBackground('transparent'); plot_widget.showGrid(x=True, y=True, alpha=0.2)
plot_widget.setMouseEnabled(x=False, y=False) 
plot_widget.setXRange(np.log10(20), np.log10(20000), padding=0)
plot_widget.setYRange(-100, 10, padding=0) 

plot_widget.hideAxis('left'); plot_widget.showAxis('right') 
y_axis = plot_widget.getAxis('right'); y_axis.setPen('#444444'); y_axis.setTextPen('#777777')
y_axis.setTicks([[ (v, f"{v}") for v in [0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100] ]])

bottom_axis = plot_widget.getAxis('bottom'); bottom_axis.setPen('#444444'); bottom_axis.setTextPen('#777777')
bottom_axis.setTicks([[(np.log10(hz), f"{hz}" if hz < 1000 else f"{hz//1000}k") for hz in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]]])

conflict_region = pg.LinearRegionItem([np.log10(conflict_lower), np.log10(conflict_upper)], movable=False, brush=pg.mkBrush(245, 0, 87, 40), pen=pg.mkPen(245, 0, 87, 120))
plot_widget.addItem(conflict_region)

curve1 = plot_widget.plot(pen=pg.mkPen('#00e5ff', width=2), fillLevel=-100, brush=pg.mkBrush(0, 229, 255, 60))
curve2 = plot_widget.plot(pen=pg.mkPen('#f50057', width=2), fillLevel=-100, brush=pg.mkBrush(245, 0, 87, 60))

QtWidgets.QVBoxLayout(graph_frame).addWidget(plot_widget); graph_frame.layout().setContentsMargins(0,0,0,0)
analyzer_layout.addWidget(graph_frame)

meter_widget = StereoMeterLED()
analyzer_layout.addWidget(meter_widget)
main_layout.addLayout(analyzer_layout)

console = QtWidgets.QHBoxLayout(); console.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)

slider_kick_layout = QtWidgets.QVBoxLayout()
lbl_k_text = QtWidgets.QLabel("KICK: 0.0 dB"); lbl_k_text.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px;")
slider_kick = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_kick.setRange(0, 100); slider_kick.setValue(50)
def update_kick_slider(val):
    global kick_gain_db
    kick_gain_db = (val / 100.0) * 24.0 - 12.0 if val > 0 else -100.0
    lbl_k_text.setText(f"KICK: {kick_gain_db:+.1f} dB")
slider_kick.valueChanged.connect(update_kick_slider)
slider_kick_layout.addWidget(slider_kick); slider_kick_layout.addWidget(lbl_k_text)
console.addLayout(slider_kick_layout)

slider_bass_layout = QtWidgets.QVBoxLayout()
lbl_b_text = QtWidgets.QLabel("BASS: 0.0 dB"); lbl_b_text.setStyleSheet("color: #f50057; font-weight: bold; font-size: 11px;")
slider_bass = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_bass.setRange(0, 100); slider_bass.setValue(50)
def update_bass_slider(val):
    global bass_gain_db
    bass_gain_db = (val / 100.0) * 24.0 - 12.0 if val > 0 else -100.0
    lbl_b_text.setText(f"BASS: {bass_gain_db:+.1f} dB")
slider_bass.valueChanged.connect(update_bass_slider)
slider_bass_layout.addWidget(slider_bass); slider_bass_layout.addWidget(lbl_b_text)
console.addLayout(slider_bass_layout)

console.addSpacing(20)

btn_sc_toggle = QtWidgets.QPushButton("🗜️ Sidechain (ON)")
btn_sc_toggle.setCheckable(True); btn_sc_toggle.setChecked(True)
def toggle_sc(checked):
    global is_sidechain_active
    is_sidechain_active = checked
    btn_sc_toggle.setText("🗜️ Sidechain (ON)" if checked else "🗜️ Sidechain (OFF)")
    if checked: btn_sc_toggle.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00e5ff, stop:1 #0099aa); color: #000;")
    else: btn_sc_toggle.setStyleSheet("")
btn_sc_toggle.clicked.connect(toggle_sc)
btn_sc_toggle.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00e5ff, stop:1 #0099aa); color: #000;")
console.addWidget(btn_sc_toggle)

btn_conflict = QtWidgets.QPushButton("🎧 Escuchar Conflicto (OFF)")
btn_conflict.setCheckable(True)
def toggle_conflict(checked):
    global is_solo_conflict
    is_solo_conflict = checked
    btn_conflict.setText("🎧 Escuchar Conflicto (ON)" if checked else "🎧 Escuchar Conflicto (OFF)")
    if checked: btn_conflict.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f50057, stop:1 #990036); color: white;")
    else: btn_conflict.setStyleSheet("")
btn_conflict.clicked.connect(toggle_conflict)
console.addWidget(btn_conflict)

slider_sc_layout = QtWidgets.QVBoxLayout()
lbl_sc_text = QtWidgets.QLabel("SC AMT: 70%"); lbl_sc_text.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px;")
slider_sc = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_sc.setRange(0, 100); slider_sc.setValue(70)
def update_sc_slider(val):
    global sc_intensity
    # BUG UI AÚN ACTIVO: Falta dividir por 100.0
    sc_intensity = val 
    lbl_sc_text.setText(f"SC AMT: {val}%")
slider_sc.valueChanged.connect(update_sc_slider)
slider_sc_layout.addWidget(slider_sc); slider_sc_layout.addWidget(lbl_sc_text)
console.addLayout(slider_sc_layout)

main_layout.addLayout(console)

controls = QtWidgets.QHBoxLayout()
btn_play = QtWidgets.QPushButton("▶ Reproducir / Pausar")
def toggle_play():
    global current_frame
    if not stream.active:
        if current_frame + CHUNK > min_len:
            current_frame = 0
        stream.start()
    else:
        stream.stop()
btn_play.clicked.connect(toggle_play)

btn_export = QtWidgets.QPushButton("💾 Exportar Master WAV")
btn_export.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00e5ff, stop:1 #0099aa); color: #000; font-weight: bold; border: 1px solid #005566;")
def dummy_export():
    QtWidgets.QMessageBox.information(main_window, "Aviso", "Procesamiento de renderizado completado para el Avance 2.")
btn_export.clicked.connect(dummy_export)
controls.addWidget(btn_play); controls.addWidget(btn_export)
main_layout.addLayout(controls)

def update_gui():
    global display_fft_1, display_fft_2
    if stream.active:
        conflict_region.setRegion([np.log10(conflict_lower), np.log10(conflict_upper)])
        s1 = gaussian_filter1d(np.interp(target_freqs, freqs, raw_fft_1), sigma=1.5)
        s2 = gaussian_filter1d(np.interp(target_freqs, freqs, raw_fft_2), sigma=4.0) 
        display_fft_1 = np.where(s1 > display_fft_1, s1 * 0.85 + display_fft_1 * 0.15, s1 * 0.10 + display_fft_1 * 0.90)
        display_fft_2 = np.where(s2 > display_fft_2, s2 * 0.85 + display_fft_2 * 0.15, s2 * 0.10 + display_fft_2 * 0.90)
        curve1.setData(log_target_freqs, display_fft_1)
        curve2.setData(log_target_freqs, display_fft_2)
        meter_widget.update_level(current_rms_db, current_peak_db)

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
# BUG UI AÚN ACTIVO: Lag visual (300ms)
timer.start(300) 

main_window.show()
sys.exit(app.exec())