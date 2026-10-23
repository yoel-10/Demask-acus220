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

# --- 1. CONFIGURACIÓN Y AUDIO BASE ---
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

# --- MOTOR DSP PROFESIONAL: DYNAMIC EQ & ENVOLVENTE LIBRE DE DISTORSIÓN ---
conflict_lower = 60.0
conflict_upper = 150.0

# Filtros y estados independientes (evita clics entre bloques CHUNK)
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

# --- WIDGET DE VÚMETRO LED ---
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

# --- HILO DE AUDIO DSP ACTIVO (DYNAMIC EQ TRANSPARENTE) ---
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