# Reconocedor de Voz con Métodos Data-Driven

---

## MIA10 - Análisis de Datos con Métodos Data-Driven

### Grupo 6

---

## Introducción

Este proyecto implementa un sistema de reconocimiento de voz utilizando métodos data-driven, sin conocimiento previo del modelo físico subyacente. El sistema identifica tanto al hablante como la palabra pronunciada mediante análisis de patrones extraídos directamente de las señales de audio.

El objetivo es demostrar cómo las técnicas de descomposición modal y análisis espectral pueden revelar estructuras coherentes en datos de voz sin modelos fisiológicos del aparato vocal humano.

## Arquitectura del Sistema

El proyecto está dividido en tres componentes principales:

- **Frontend Web**: Interfaz de usuario para grabación y visualización
- **Backend Python**: Motor de procesamiento con algoritmos data-driven
- **Almacenamiento**: Gestión de muestras de entrenamiento y modelos

El flujo básico consiste en: grabación → extracción de características → clasificación → predicción.

## Métodos Data-Driven Implementados

### 1. Transformada Rápida de Fourier (FFT)

**Justificación:**
La FFT permite analizar el contenido frecuencial de la señal de voz sin requerir conocimiento del modelo físico del aparato vocal.

**Implementación:**
```python
def _extract_fft_features(self, audio_data, sample_rate):
    windowed = audio_data * signal.windows.hamming(len(audio_data))
    fft_result = fft(windowed, n=self.fft_size)
    magnitude_spectrum = np.abs(fft_result[:self.fft_size//2])
```

**Características extraídas:**
- Centroide espectral: Centro de masa del espectro de frecuencias
- Rolloff espectral: Frecuencia por debajo de la cual se concentra el 85% de la energía
- Ancho de banda espectral: Dispersión del espectro alrededor del centroide
- Tasa de cruces por cero: Indicador de la periodicidad de la señal
- 5 frecuencias dominantes: Picos más prominentes en el espectro

**Justificación técnica:**
Cada persona tiene características vocales únicas que se reflejan en patrones frecuenciales específicos. Las cuerdas vocales, cavidades resonantes y articulación crean "huellas espectrales" distintivas que la FFT puede capturar sin modelar explícitamente la anatomía vocal.

### 2. Coeficientes Cepstrales en Escala Mel (MFCC)

**¿Por qué MFCC?**
Los MFCC simulan la percepción auditiva humana mediante la escala mel, que es logarítmica y se ajusta mejor a cómo procesamos frecuencias. Esto nos da una representación más "natural" del audio.

**Implementación:**
```python
def _extract_mfcc_features(self, audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=self.mfcc_n_coeffs,
        n_fft=self.fft_size, hop_length=self.hop_length, n_mels=self.n_mels
    )
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
```

**Proceso detallado:**
1. Aplicación de filtros mel triangulares al espectrograma
2. Logaritmo de la energía en cada banda mel
3. Transformada discreta del coseno (DCT) para decorrelación
4. Extracción de media y desviación estándar temporal

**Características extraídas:**
- 13 coeficientes MFCC promedio
- 13 desviaciones estándar de los MFCC
- Total: 26 características

**Justificación técnica:**
Los MFCC capturan la envolvente espectral que caracteriza los fonemas y la identidad vocal. La escala mel reduce la redundancia en altas frecuencias y enfatiza la información perceptualmente relevante.

### 3. Descomposición Modal Dinámica (DMD)

**¿Por qué DMD?**
DMD es fundamental para este proyecto porque puede extraer modos dinámicos coherentes de datos temporales sin conocer las ecuaciones que gobiernan el sistema. Trata las señales de voz como un sistema dinámico no lineal.

**Implementación:**
```python
def _extract_dmd_features(self, audio_data, sample_rate):
    # Crear matrices de snapshots temporales
    X1 = X[:, :-1]  # Snapshots 1 a n-1
    X2 = X[:, 1:]   # Snapshots 2 a n

    # SVD de X1
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)

    # Construir operador reducido
    Atilde = U_r.T @ X2 @ V_r @ np.diag(1/S_r)

    # Eigendescomposición
    eigenvals, eigenvecs = np.linalg.eig(Atilde)
```

**Fundamento matemático:**
DMD asume que existe un operador lineal A tal que X₂ = AX₁, donde X₁ y X₂ son matrices de datos consecutivos. Los eigenvalores de A revelan las frecuencias y tasas de crecimiento/decaimiento de los modos dinámicos.

**Características extraídas:**
- 5 frecuencias dominantes de los modos
- 5 tasas de crecimiento/decaimiento correspondientes
- Total: 10 características

**Justificación técnica:**
La voz humana exhibe dinámicas cuasi-periódicas y transitorias que DMD puede separar sin conocimiento a priori. Los eigenvalores complejos codifican tanto la frecuencia fundamental como los armónicos, mientras que sus magnitudes indican la estabilidad temporal de cada modo.

### 4. Descomposición en Valores Singulares (SVD)

**¿Por qué SVD?**
SVD descompone el espectrograma en componentes principales ordenados por importancia energética. Es equivalente a PCA pero aplicado directamente a la representación tiempo-frecuencia.

**Implementación:**
```python
def _extract_svd_features(self, audio_data, sample_rate):
    f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate)
    log_spec = np.log(Sxx + 1e-10)
    U, S, Vt = np.linalg.svd(log_spec, full_matrices=False)
```

**Proceso de extracción:**
1. Cálculo del espectrograma log-magnitud
2. SVD: log_spec = USVᵀ
3. Análisis de la distribución energética en valores singulares
4. Cálculo de métricas de complejidad espectral

**Características extraídas:**
- 8 primeros ratios de energía singular
- Entropía espectral (complejidad)
- Rango efectivo (diversidad espectral)
- Total: 10 características

**Justificación técnica:**
SVD revela la estructura de bajo rango en espectrogramas de voz. Los primeros valores singulares capturan los patrones dominantes (formato de voz, pitch), mientras que la distribución energética caracteriza la riqueza tímbrica individual.

### 5. Características Estadísticas Temporales

**¿Por qué estadísticas temporales?**
Complementan el análisis frecuencial con información sobre la envolvente temporal y dinámica de amplitud, capturando patrones de prosodia y articulación.

**Características extraídas:**
- Media temporal
- Desviación estándar
- Amplitud máxima
- Energía RMS
- Asimetría (skewness)
- Curtosis (kurtosis)

**Justificación técnica:**
Los momentos estadísticos de orden superior (skewness, kurtosis) caracterizan la distribución de amplitudes de manera robusta y son sensibles a patrones de articulación y respiración individuales.

## Estrategia de Clasificación

### Enfoque Ensemble

Utilizamos dos clasificadores especializados:

**Random Forest para identificación de usuario:**
- Robusto ante características ruidosas
- Maneja bien la alta dimensionalidad (61 características)
- Proporciona importancia de características

**SVM con kernel RBF para reconocimiento de palabras:**
- Efectivo para separar patrones complejos no lineales
- Kernel RBF captura relaciones locales entre características
- Probabilidades calibradas para confianza

### Preprocesamiento

1. **Normalización estándar**: Cada característica se escala a media 0 y desviación 1
2. **Codificación de etiquetas**: Usuarios y palabras se mapean a índices numéricos
3. **Validación cruzada**: Evaluación robusta del rendimiento

## Fundamento Teórico

### Principio Data-Driven

El enfoque data-driven se basa en la hipótesis de que las señales de voz contienen suficiente información estructurada para permitir clasificación sin modelos físicos explícitos. Esto contrasta con enfoques tradicionales que modelan:

- Anatomía del tracto vocal
- Mecánica de las cuerdas vocales
- Acústica de cavidades resonantes

### Teoría del Operador de Koopman

DMD aproxima el operador de Koopman, que lineariza dinámicas no lineales en un espacio de observables. Para señales de voz:

```
z(t+1) = K z(t)
```

donde z(t) representa el estado del sistema vocal y K es el operador de Koopman. Los eigenvalores de K revelan modos intrínsecos del sistema.

### Análisis Multimodal

La combinación de métodos espectrales (FFT, MFCC), temporales (DMD) y espaciales (SVD) proporciona una caracterización completa que captura:

- **Contenido frecuencial**: Formantes y armónicos
- **Dinámicas temporales**: Transitorios y periodicidades
- **Estructura espacio-temporal**: Patrones emergentes en espectrogramas

## Configuración y Ejecución

### Instalación

```bash
cd TP2_ReconocedorVoz
pip install -r requirements.txt
```

### Ejecución del Backend

```bash
cd backend
python app.py
```
El servidor Flask se iniciará en `http://localhost:8000`

### Interfaz Web

La aplicación incluye interfaz web integrada con grabación en tiempo real, visualizaciones de FFT, MFCC, DMD y SVD, métricas de rendimiento y panel de evaluación del modelo.

## Protocolo Experimental

### Fase de Entrenamiento

1. Grabar 3-5 muestras por usuario/palabra
2. Duración recomendada: 2-3 segundos por muestra
3. Ambiente: Minimizar ruido de fondo
4. Consistencia: Mantener distancia y orientación al micrófono

### Fase de Evaluación

1. Grabar muestras de prueba no incluidas en entrenamiento
2. Evaluar precisión por usuario y por palabra
3. Analizar matriz de confusión para identificar errores sistemáticos
4. Calcular confianza de predicciones

## Limitaciones y Consideraciones

### Limitaciones Actuales

- Vocabulario limitado (5 palabras)
- Sensible a ruido de fondo
- Requiere condiciones de grabación consistentes
- Número limitado de usuarios simultáneos

### Extensiones Posibles

- Implementación de HODMD para capturar dinámicas de mayor orden
- Uso de mrDMD para análisis multi-resolución temporal
- Incorporación de características prosódicas (entonación, ritmo)
- Técnicas de aumento de datos para robustez

## Referencias Técnicas

- Schmid, P.J. (2010). "Dynamic Mode Decomposition of Numerical and Experimental Data"
- Kutz, J.N. et al. (2016). "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems"
- Le Clainche, S. & Vega, J.M. (2017). "Higher-order dynamic mode decomposition"
- Davis, S. & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition"

## Resultados

### Rendimiento
El sistema procesa audio en tiempo real con tiempos de respuesta menores a 50ms para predicción. La extracción de 61 características se completa en aproximadamente 30ms.

### Precisión
Los experimentos muestran precisión variable según el tamaño del dataset: con validación cruzada se obtiene entre 70-90% de precisión tanto para usuarios como para palabras. Con datasets pequeños la precisión se penaliza para evitar sobreajuste.

### Implementación
Se utilizan los métodos especificados: FFT para análisis espectral, MFCC para características perceptuales, DMD para dinámicas temporales según Schmid (2010), y SVD para descomposición de espectrogramas. El sistema incluye interfaz web para las pruebas.

El proyecto demuestra la efectividad de métodos data-driven para reconocimiento de voz, revelando la información contenida en señales acústicas cuando se analizan con herramientas matemáticas apropiadas, sin conocimiento físico explícito del sistema vocal humano.