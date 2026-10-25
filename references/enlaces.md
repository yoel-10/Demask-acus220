## Bibliografía y Enlaces Útiles - Proyecto DeMask

### 1. Documentación de Librerías (Python)
Herramientas principales para el desarrollo del motor DSP y la interfaz:
* **SciPy (scipy.fft):** Transformada Rápida de Fourier discreta (FFT) y ventaneo (mitigar fuga espectral).
* **SciPy (scipy.signal):** Diseño y aplicación de filtros Biquad para el aislamiento de resonancias.
* **Sounddevice:** Lectura simultánea y enrutamiento de audio en tiempo real con control de tamaño de buffer.
* **PyQtGraph:** Representación visual de espectrogramas (FFT) y Vúmetros en tiempo real optimizando el uso de CPU.

### 2. Procesamiento Digital de Señales (DSP)
Material teórico que respalda las decisiones algorítmicas de la ecualización dinámica:
* **Filtros Biquad (Audio EQ Cookbook - Robert Bristow-Johnson):** Fórmulas estándar de la industria para el diseño de filtros paramétricos transparentes.
* **Detección RMS y Envolventes:** *DAFX - Digital Audio Effects (Udo Zölzer)*. Conceptos de Attack/Release para compresión y reducción de ganancia libre de distorsión.
* **Julius O. Smith III (CCRMA, Stanford):** Apuntes clásicos sobre filtros digitales y matemáticas de la transformada discreta de Fourier (DFT).

### 3. Psicoacústica y Percepción Auditiva
Textos que justifican el análisis del choque de frecuencias y la reducción de ganancia logarítmica:
* **Psicoacústica y Escalas Logarítmicas:** *Psychoacoustics: Facts and Models (Zwicker & Fastl)*. Justificación del procesamiento de volumen en decibelios (dB) para imitar la percepción humana.
* **HyperPhysics (Georgia State University):** Conceptos sobre bandas críticas y enmascaramiento auditivo en el "Low-End".