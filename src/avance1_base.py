import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d

# 1. CARGA DE ARCHIVOS Y PARÁMETROS BÁSICOS
archivo_kick = "data/raw/kick.wav"
archivo_bajo = "data/raw/Bajo.wav"
CHUNK = 4096  # Tamaño del buffer de audio

def cargar_audios(path_kick, path_bajo):
    try:
        data_k, fs = sf.read(path_kick, dtype='float32')
        data_b, _ = sf.read(path_bajo, dtype='float32')

        # Convertir a mono si vienen en estéreo
        if data_k.ndim > 1:
            data_k = data_k[:, 0]
        if data_b.ndim > 1:
            data_b = data_b[:, 0]

        # Ajustar ambos al mismo largo
        largo_min = min(len(data_k), len(data_b))
        data_k = data_k[:largo_min]
        data_b = data_b[:largo_min]

        return data_k, data_b, fs, largo_min
    except Exception as error:
        print(f"Error al leer archivos: {error}")
        return np.zeros(CHUNK), np.zeros(CHUNK), 44100, CHUNK

audio_kick, audio_bajo, fs, total_muestras = cargar_audios(archivo_kick, archivo_bajo)

# Variables de control de reproducción y niveles
indice_reproduccion = 0
ganancia_kick_db = 0.0
ganancia_bajo_db = 0.0
rms_actual = -100.0
peak_actual = -100.0

# Preparación para cálculo de FFT y escala logarítmica
num_bins = CHUNK // 2 + 1
fft_kick_raw = np.full(num_bins, -100.0)
fft_bajo_raw = np.full(num_bins, -100.0)

frecuencias = rfftfreq(CHUNK, 1.0 / fs)
frecuencias[0] = 1.0  # Evita log(0)

puntos_grafico = 500
eje_frecuencias_log = np.logspace(np.log10(20), np.log10(20000), puntos_grafico)
eje_x_grafico = np.log10(eje_frecuencias_log)

curva_kick_suave = np.full(puntos_grafico, -100.0)
curva_bajo_suave = np.full(puntos_grafico, -100.0)

# 2. PROCESAMIENTO DE AUDIO EN TIEMPO REAL
def procesar_audio(outdata, frames, time_info, status):
    global indice_reproduccion, fft_kick_raw, fft_bajo_raw, rms_actual, peak_actual

    # Comprobar si se terminó el audio
    if indice_reproduccion + frames > total_muestras:
        outdata.fill(0)
        raise sd.CallbackStop

    # Extraer el bloque actual
    bloque_k = audio_kick[indice_reproduccion : indice_reproduccion + frames].copy()
    bloque_b = audio_bajo[indice_reproduccion : indice_reproduccion + frames].copy()

    # Aplicar volumen (dB a factor lineal)
    if ganancia_kick_db <= -99.0:
        factor_k = 0.0
    else:
        factor_k = 10 ** (ganancia_kick_db / 20.0)

    if ganancia_bajo_db <= -99.0:
        factor_b = 0.0
    else:
        factor_b = 10 ** (ganancia_bajo_db / 20.0)

    bloque_k = bloque_k * factor_k
    bloque_b = bloque_b * factor_b

    # Mezcla estéreo básica (señal mono centrada)
    mezcla = (bloque_k + bloque_b) * 0.5
    outdata[:, 0] = mezcla
    outdata[:, 1] = mezcla

    # Telemetría de niveles
    peak_actual = 20 * np.log10(np.max(np.abs(mezcla)) + 1e-10)
    rms_actual = 20 * np.log10(np.sqrt(np.mean(mezcla**2)) + 1e-10)

    # Cálculo de FFT con ventana Hanning
    ventana = np.hanning(frames)
    fft_k = np.abs(rfft(bloque_k * ventana)) / (CHUNK / 2) * 2.0
    fft_b = np.abs(rfft(bloque_b * ventana)) / (CHUNK / 2) * 2.0

    fft_kick_raw = 20 * np.log10(np.clip(fft_k, 1e-10, 1.0))
    fft_bajo_raw = 20 * np.log10(np.clip(fft_b, 1e-10, 1.0))

    indice_reproduccion += frames

stream = sd.OutputStream(samplerate=fs, channels=2, blocksize=CHUNK, callback=procesar_audio)

# 3. INTERFAZ GRÁFICA (PyQt + pyqtgraph)
app = QtWidgets.QApplication(sys.argv)
ventana = QtWidgets.QWidget()
ventana.setWindowTitle("Proyecto DeMask - Análisis Kick y Bajo")
ventana.resize(1000, 680)

layout_principal = QtWidgets.QVBoxLayout(ventana)

# Título y estado
layout_encabezado = QtWidgets.QHBoxLayout()
lbl_titulo = QtWidgets.QLabel("DeMask - Analizador de Espectro")
lbl_titulo.setStyleSheet("font-size: 15px; font-weight: bold;")
lbl_estado = QtWidgets.QLabel("Estado: Detenido")

layout_encabezado.addWidget(lbl_titulo)
layout_encabezado.addStretch()
layout_encabezado.addWidget(lbl_estado)
layout_principal.addLayout(layout_encabezado)

# Gráfica espectral y panel lateral
layout_medio = QtWidgets.QHBoxLayout()

plot_widget = pg.PlotWidget()
plot_widget.showGrid(x=True, y=True, alpha=0.3)
plot_widget.setMouseEnabled(x=False, y=False)
plot_widget.setXRange(np.log10(20), np.log10(20000), padding=0)
plot_widget.setYRange(-90, 0, padding=0)

# Ejes de la gráfica
eje_x = plot_widget.getAxis('bottom')
eje_x.setLabel("Frecuencia (Hz)")
ticks_frec = [(np.log10(f), str(f)) for f in [50, 100, 200, 500, 1000, 2000, 5000, 10000]]
eje_x.setTicks([ticks_frec])

eje_y = plot_widget.getAxis('left')
eje_y.setLabel("Magnitud (dB)")

linea_kick = plot_widget.plot(pen=pg.mkPen('c', width=2), name="Kick")
linea_bajo = plot_widget.plot(pen=pg.mkPen('m', width=2), name="Bajo")
layout_medio.addWidget(plot_widget, stretch=4)

# Panel de telemetría (métricas DSP)
grupo_telemetria = QtWidgets.QGroupBox("Datos DSP")
layout_telemetria = QtWidgets.QVBoxLayout(grupo_telemetria)

lbl_sr = QtWidgets.QLabel(f"Fs: {fs} Hz")
lbl_chunk = QtWidgets.QLabel(f"Buffer: {CHUNK} muestras")
lbl_resolucion = QtWidgets.QLabel(f"Resolución: {fs / CHUNK:.2f} Hz")
lbl_rms = QtWidgets.QLabel("RMS: -100.0 dB")
lbl_peak = QtWidgets.QLabel("Peak: -100.0 dB")

for lbl in [lbl_sr, lbl_chunk, lbl_resolucion, lbl_rms, lbl_peak]:
    layout_telemetria.addWidget(lbl)
layout_telemetria.addStretch()

layout_medio.addWidget(grupo_telemetria, stretch=1)
layout_principal.addLayout(layout_medio)

# Panel inferior con controles
grupo_controles = QtWidgets.QGroupBox("Controles de Reproducción y Mezcla")
layout_controles = QtWidgets.QHBoxLayout(grupo_controles)

btn_reproducir = QtWidgets.QPushButton("Reproducir")
btn_pausar = QtWidgets.QPushButton("Pausar")
btn_reiniciar = QtWidgets.QPushButton("Reiniciar")

layout_controles.addWidget(btn_reproducir)
layout_controles.addWidget(btn_pausar)
layout_controles.addWidget(btn_reiniciar)

# Slider Ganancia Kick
lbl_slider_k = QtWidgets.QLabel("Kick: 0.0 dB")
slider_k = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_k.setRange(0, 100)
slider_k.setValue(50)

def cambio_slider_kick(valor):
    global ganancia_kick_db
    if valor == 0:
        ganancia_kick_db = -100.0
    else:
        ganancia_kick_db = (valor / 100.0) * 24.0 - 12.0
    lbl_slider_k.setText(f"Kick: {ganancia_kick_db:+.1f} dB")

slider_k.valueChanged.connect(cambio_slider_kick)

layout_slider_k = QtWidgets.QVBoxLayout()
layout_slider_k.addWidget(lbl_slider_k)
layout_slider_k.addWidget(slider_k)
layout_controles.addLayout(layout_slider_k)

# Slider Ganancia Bajo
lbl_slider_b = QtWidgets.QLabel("Bajo: 0.0 dB")
slider_b = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider_b.setRange(0, 100)
slider_b.setValue(50)

def cambio_slider_bajo(valor):
    global ganancia_bajo_db
    if valor == 0:
        ganancia_bajo_db = -100.0
    else:
        ganancia_bajo_db = (valor / 100.0) * 24.0 - 12.0
    lbl_slider_b.setText(f"Bajo: {ganancia_bajo_db:+.1f} dB")

slider_b.valueChanged.connect(cambio_slider_bajo)

layout_slider_b = QtWidgets.QVBoxLayout()
layout_slider_b.addWidget(lbl_slider_b)
layout_slider_b.addWidget(slider_b)
layout_controles.addLayout(layout_slider_b)

layout_principal.addWidget(grupo_controles)

# Funciones de botones
def accion_reproducir():
    stream.start()
    lbl_estado.setText("Estado: Reproduciendo")

def accion_pausar():
    stream.stop()
    lbl_estado.setText("Estado: Pausado")

def accion_reiniciar():
    global indice_reproduccion
    indice_reproduccion = 0
    if not stream.active:
        stream.start()
    lbl_estado.setText("Estado: Reiniciado")

btn_reproducir.clicked.connect(accion_reproducir)
btn_pausar.clicked.connect(accion_pausar)
btn_reiniciar.clicked.connect(accion_reiniciar)

# 4. TEMPORIZADOR DE ACTUALIZACIÓN VISUAL
def actualizar_grafico():
    global curva_kick_suave, curva_bajo_suave

    if stream.active:
        # Interpolación a escala logarítmica y filtro gaussiano
        k_interp = np.interp(eje_frecuencias_log, frecuencias, fft_kick_raw)
        b_interp = np.interp(eje_frecuencias_log, frecuencias, fft_bajo_raw)

        s_kick = gaussian_filter1d(k_interp, sigma=1.5)
        s_bajo = gaussian_filter1d(b_interp, sigma=4.0)

        # Suavizado temporal (ataque y decaimiento)
        curva_kick_suave = np.where(
            s_kick > curva_kick_suave,
            s_kick * 0.85 + curva_kick_suave * 0.15,
            s_kick * 0.10 + curva_kick_suave * 0.90
        )
        curva_bajo_suave = np.where(
            s_bajo > curva_bajo_suave,
            s_bajo * 0.85 + curva_bajo_suave * 0.15,
            s_bajo * 0.10 + curva_bajo_suave * 0.90
        )

        linea_kick.setData(eje_x_grafico, curva_kick_suave)
        linea_bajo.setData(eje_x_grafico, curva_bajo_suave)

        lbl_rms.setText(f"RMS: {rms_actual:.1f} dB")
        lbl_peak.setText(f"Peak: {peak_actual:.1f} dB")

timer = QtCore.QTimer()
timer.timeout.connect(actualizar_grafico)
timer.start(30)  # ~33 FPS

# 5. EJECUCIÓN DE LA APLICACIÓN
ventana.show()
sys.exit(app.exec())