# Bibliografía y Enlaces Útiles - Proyecto DeMask

## 1. Documentación de Librerías (Python)
Herramientas principales que estoy usando para armar el código del DSP (Fases 1 y 2):

*   **SciPy (scipy.fft):** Para calcular la Transformada Rápida de Fourier discreta y aplicar el ventaneo (mitigar el spectral leakage).
    *   [Docs scipy.fft](https://docs.scipy.org/doc/scipy/reference/fft.html)
*   **SciPy (scipy.signal):** Para el diseño y aplicación de los filtros IIR (Butterworth / Linkwitz-Riley) que separarán las bandas.
    *   [Docs scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
*   **Sounddevice:** Para la lectura de los audios simultáneos controlando el tamaño del buffer y la latencia.
    *   [Docs python-sounddevice](https://python-sounddevice.readthedocs.io/)
*   **PyQtGraph:** Para graficar la telemetría y las FFT en tiempo real optimizando el uso de CPU.
    *   [Docs pyqtgraph](https://pyqtgraph.readthedocs.io/en/latest/)

## 2. Procesamiento Digital de Señales (DSP)
Material teórico para respaldar la matemática detrás de los filtros y el análisis:

*   **Julius O. Smith III (CCRMA, Stanford):** Apuntes clásicos sobre filtros digitales y análisis espectral. 
    *   *Mathematics of the Discrete Fourier Transform (DFT)*: [Libro online](https://ccrma.stanford.edu/~jos/mdft/)
    *   *Introduction to Digital Filters*: [Libro online](https://ccrma.stanford.edu/~jos/filters/)
*   **Linkwitz Lab:** Teoría base sobre la topología Linkwitz-Riley para evitar problemas de fase en el cruce de frecuencias.
    *   [Crossover Topology](https://www.linkwitzlab.com/crossovers.htm)

## 3. Psicoacústica y Enmascaramiento
Textos para justificar el conflicto en el Low-End y cómo lo percibe el oído humano:

*   **HyperPhysics (Georgia State University):** Resumen de conceptos sobre bandas críticas y enmascaramiento auditivo.
    *   [Auditory Masking](http://hyperphysics.phy-astr.gsu.edu/hbase/Sound/mask.html)