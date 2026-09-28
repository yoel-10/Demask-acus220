# DeMask - Smart Dynamic EQ

## Integrantes
- Yoel Ezequiel Nuñez Avilés
- **Carrera:** Ingeniería Civil Acústica
- **Asignatura:** ACUS220 — Acústica Computacional con Python

## Idea del proyecto
Desarrollar un plugin DSP de asistencia de mezcla enfocado en detectar y corregir automáticamente el enmascaramiento frecuencial entre pistas de bombo (Kick) y bajo (Bass).

## Pregunta principal
¿Es posible automatizar la detección de conflictos de fase y frecuencia en el rango de subgraves mediante análisis FFT para aplicar un ducking dinámico quirúrgico sin afectar la integridad del arreglo musical?

## Motivación
El choque de graves es uno de los problemas más críticos en géneros como el Mambo Urbano, Reggaetón y Trap. El objetivo es crear una solución técnica que reemplace el sidechain estático tradicional, permitiendo que el bajo mantenga su peso y presencia mientras se abre espacio dinámico para el transitorio del bombo.

## Datos
- Fuente: Archivos WAV crudos (stems de sesiones de mezcla).
- Tipo de datos: Señales de audio PCM sin comprimir.
- Formato: .wav (Mono/Stereo, 44.1kHz - 48kHz, 24-bit).
- Cantidad aproximada: Archivos de hasta 40 MB por pista.
- Aspectos que todavía debemos investigar: Optimización del cálculo FFT en tiempo real con Python y diseño de filtros Biquad para corrección paramétrica.

## Alcance inicial
Construir un prototipo funcional que logre cargar dos pistas WAV, analizarlas espectralmente y visualizar la zona de choque en una interfaz gráfica básica utilizando PyQt y PyQtGraph.

## Pipeline provisional
Datos
-> Carga de arrays de audio (Soundfile)
-> Análisis de Fourier (SciPy rfft)
-> Visualización de espectro superpuesto

## Posibles dificultades
- Latencia en el renderizado del gráfico a altas tasas de refresco (FPS).
- Consumo excesivo de CPU al calcular la FFT en bloques grandes (Chunks).

## Estado actual
Definición de la arquitectura base e investigación de librerías DSP en Python.

## Próximos pasos
1. Programar el motor de carga y reproducción sincrónica de audio.
2. Levantar la interfaz gráfica inicial con PyQtGraph.
3. Implementar el cálculo FFT y el detector visual de solapamiento.

## Organización del Repositorio
* `data/raw/`: Almacena los stems de audio (.wav) originales para las pruebas (aislando el Kick y Bass).
* `src/`: Contiene el código fuente en Python (motor de audio, cálculo DSP y renderizado PyQtGraph).
* `references/`: Incluye bibliografía, enlaces a documentación de librerías y papers científicos sobre acústica.
* `figures/`: Capturas de pantalla de la interfaz gráfica y evidencias de avance.