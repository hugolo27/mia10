@echo off
echo 🎤 Iniciando Reconocedor de Voz - TP2 MIA10
echo ==========================================

REM Verificar si Python está instalado
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Python no está instalado
    echo    Instalar Python 3.8+ desde https://python.org
    pause
    exit /b 1
)

echo ✅ Python encontrado
python --version

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Error creando entorno virtual
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias
echo 📚 Instalando dependencias...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error instalando dependencias
    pause
    exit /b 1
)

echo 🚀 Iniciando servidor backend en puerto 8000...

REM Iniciar servidor en background
start /B python backend/app.py

REM Esperar que el servidor se inicie
timeout /t 3 /nobreak >nul

REM Verificar que el servidor esté corriendo
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servidor backend iniciado correctamente
    echo 🌐 Abriendo interfaz web...

    REM Abrir navegador
    start frontend/index.html

    echo.
    echo 🎯 SISTEMA LISTO!
    echo    Backend: http://localhost:8000
    echo    Frontend: Abierto en navegador
    echo.
    echo 📖 Para usar:
    echo    1. Entrenar con tu voz (3-5 muestras por palabra)
    echo    2. Probar reconocimiento
    echo.
    echo ⏹️  Para detener: Cerrar esta ventana

    REM Mantener la ventana abierta
    pause

) else (
    echo ❌ Error: Servidor no pudo iniciarse
    echo    Verificar puerto 8000 disponible
    pause
    exit /b 1
)