# DeMask: Sidechain Inteligente y Análisis Espectral

**Asignatura:** ACUS220 — Acústica Computacional con Python  
**Estudiante:** Yoel Ezequiel Núñez Avilés  
**Universidad:** Universidad Austral de Chile (Facultad de Ciencias de la Ingeniería)  

## Resumen del Proyecto
DeMask es un motor de procesamiento de audio en tiempo real diseñado para solucionar problemas de enmascaramiento frecuencial (frecuency masking) en el "Low-End", específicamente entre el bombo (Kick) y el bajo (Bass). Utiliza análisis espectral para aislar la frecuencia de conflicto y aplica ecualización dinámica (sidechain sustractivo) de forma acústicamente transparente.

## Hito 2: Motor DSP y Ecualización Dinámica (Octubre 2026)
Durante esta fase, el proyecto evolucionó de un simple analizador visual a un procesador de señal digital (DSP) funcional en tiempo real. 

**Principales desafíos y soluciones implementadas:**
* **Filtros Biquad Coherentes:** Se descartó el uso de separadores Crossover estáticos (que causaban cancelación de fase destructiva) en favor de Filtros Biquad pasa-banda, aislando matemáticamente solo la resonancia problemática.
* **Detección RMS de Dos Etapas:** Para evitar una respuesta brusca (distorsión tipo "temblor"), se implementó un sistema de lectura de energía con tiempos de ataque (18 ms) y relajación (65 ms) mediante filtros envolventes (Envelope Detection).
* **Interfaz Gráfica (GUI):** Desarrollada con `pyqtgraph`, operando a 30ms para evitar sobrecargas del procesador. Incluye analizador de espectro en vivo, controles de ganancia por decibelios (dB), control de intensidad de Sidechain y un Vúmetro LED integrado.

## Estructura del Repositorio
* `src/` : Contiene el código fuente principal (`avance2_motor.py`).
* `data/raw/` : Pistas de audio de prueba (`Kick.wav` y `Bass.wav`).
* `doc/` : Documentación y avances del Hito 1.
* `references/` : Bibliografía y material teórico en formato PDF (`DAFX`, `Zwicker & Fastl`, etc.).

## Instalación y Uso
Para ejecutar el motor DeMask, asegúrate de tener instaladas las dependencias del proyecto:

```bash
pip install numpy scipy sounddevice soundfile pyqtgraph PyQt6