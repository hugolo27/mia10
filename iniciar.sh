#!/bin/bash

echo "🎤 Iniciando Reconocedor de Voz - TP2 MIA10"
echo "=========================================="

# Verificar si Python está instalado
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: Python no está instalado"
        echo "   Instalar Python 3.8+ desde https://python.org"
        exit 1
    else
        alias python=python3
    fi
fi

echo "✅ Python encontrado: $(python --version)"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error creando entorno virtual"
        exit 1
    fi
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Verificar si pip está instalado
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip no está disponible"
    exit 1
fi

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "❌ Error instalando dependencias"
    exit 1
fi

echo "🚀 Iniciando servidor backend en puerto 8000..."
echo "   Abriendo en segundo plano..."

# Iniciar servidor en background
python backend/app.py &
SERVER_PID=$!

# Esperar que el servidor se inicie
sleep 3

# Verificar que el servidor esté corriendo
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Servidor backend iniciado correctamente"
    echo "🌐 Abriendo interfaz web..."

    # Abrir navegador
    if command -v open &> /dev/null; then
        # macOS
        open frontend/index.html
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open frontend/index.html
    elif command -v start &> /dev/null; then
        # Windows
        start frontend/index.html
    else
        echo "📂 Abrir manualmente: frontend/index.html"
    fi

    echo ""
    echo "🎯 SISTEMA LISTO!"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: Abierto en navegador"
    echo ""
    echo "📖 Para usar:"
    echo "   1. Entrenar con tu voz (3-5 muestras por palabra)"
    echo "   2. Probar reconocimiento"
    echo ""
    echo "⏹️  Para detener: Ctrl+C"

    # Mantener el script corriendo
    wait $SERVER_PID

else
    echo "❌ Error: Servidor no pudo iniciarse"
    echo "   Verificar puerto 8000 disponible"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi