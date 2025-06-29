# 📖 Guía de Desarrollo

Esta guía está destinada a los desarrolladores que deseen contribuir al proyecto KUMAK. Proporciona información sobre cómo configurar el entorno de desarrollo, ejecutar pruebas y seguir las convenciones de código.

## 🛠️ Configuración del Entorno

1.  **Clonar el Repositorio**: `git clone https://github.com/tu-usuario/kumak.git`
2.  **Entorno Virtual**: Se recomienda usar `venv` para gestionar las dependencias:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Instalar Dependencias**: Usa `pip` o `poetry` para instalar las dependencias del proyecto:
    ```bash
    pip install -r requirements.txt
    # O si usas Poetry:
    poetry install
    ```

## ✅ Pruebas

El proyecto incluye un conjunto de pruebas para garantizar la calidad y el correcto funcionamiento del código. Para ejecutar las pruebas, utiliza los siguientes comandos:

```bash
# Pruebas unitarias y de integración
python -m unittest discover tests

# Simulación de un flujo de conversación
python tests/simulate_business_context.py
```

## 🎨 Estilo de Código y Convenciones

- **Formato de Código**: Se utiliza `black` para el formateo automático del código. Asegúrate de ejecutarlo antes de hacer un commit:
  ```bash
  black .
  ```
- **Análisis Estático**: `ruff` se utiliza para el análisis de código y la detección de errores comunes. Es recomendable integrarlo en tu editor de código.
- **Convenciones de Nomenclatura**: Sigue las convenciones de PEP 8 para nombrar variables, funciones y clases.

## 📦 Gestión de Dependencias

El proyecto utiliza `poetry` para la gestión de dependencias. Si necesitas añadir una nueva dependencia, usa el siguiente comando:

```bash
poetry add <nombre_del_paquete>
```

Esto actualizará los archivos `pyproject.toml` y `poetry.lock`.