import streamlit as st
import pandas as pd
import plotly.express as px
import io
import re
import json
import requests
import time # Para implementar la lógica de reintento (backoff)

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="NydIA: Agente de Análisis con NLP Avanzado")

# ----------------------------------------------------
# CONFIGURACIÓN DE LA API DE GEMINI
# ----------------------------------------------------
# La clave de API se obtiene del entorno de ejecución (Canvas)
API_KEY = ""
API_MODEL = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent?key={API_KEY}"
MAX_RETRIES = 5

# ----------------------------------------------------
# 1. FUNCIÓN DE PERCEPCIÓN Y CONSOLIDACIÓN (Compatibilidad total de archivos)
# ----------------------------------------------------
@st.cache_data
def consolidar_archivos(uploaded_files):
    """Procesa una lista de archivos (CSV, XLS, XLSX) y devuelve un DataFrame consolidado."""
    
    if not uploaded_files:
        return pd.DataFrame() 

    dataframes = []
    
    for file in uploaded_files:
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension in ['xls', 'xlsx']:
                # Lectura de Excel
                df = pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
            elif file_extension == 'csv':
                # Lectura de CSV: Intentamos coma (,) y luego punto y coma (;)
                file_content = io.StringIO(file.getvalue().decode('utf-8', errors='ignore'))
                
                # Intento 1: Coma como delimitador
                try:
                    df = pd.read_csv(file_content, delimiter=',', on_bad_lines='skip', encoding='utf-8')
                except Exception:
                    # Intento 2: Punto y coma como delimitador
                    file_content.seek(0) # Resetear el puntero
                    df = pd.read_csv(file_content, delimiter=';', on_bad_lines='skip', encoding='utf-8')
            else:
                st.warning(f"Tipo de archivo no soportado: {file.name}")
                continue

            dataframes.append(df)
        except Exception as e:
            st.error(f"Error al leer el archivo {file.name}: {e}")
            
    if dataframes:
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        # Intentar inferir objetos para asegurar la correcta lectura de tipos
        df_consolidado = df_consolidado.infer_objects() 
        return df_consolidado
    else:
        return pd.DataFrame()

# ----------------------------------------------------
# 2. FUNCIÓN DE COGNICIÓN: GENERACIÓN DE INSIGHTS CON GEMINI (Integración de IA)
# ----------------------------------------------------
def generar_insight_con_gemini(df_insight, eje_x, eje_y, tipo_grafico, pregunta_nlp):
    """
    Genera un insight analítico utilizando la API de Gemini.
    Envía una porción del DataFrame y el contexto de la pregunta para obtener un resumen.
    """
    if df_insight.empty:
        return "No hay datos para analizar. Por favor, ajusta los filtros."

    # 1. Preparar el contexto de los datos (solo un resumen y las primeras 5 filas)
    data_summary = f"""
    Resumen Estadístico del DataFrame filtrado (describe):\n
    {df_insight.describe(include='all').to_markdown()}

    Primeras 5 filas del DataFrame filtrado:\n
    {df_insight.head(5).to_markdown()}
    """
    
    # 2. Definir la instrucción del sistema (Persona y Formato)
    system_prompt = {
        "parts": [{
            "text": (
                "Actúa como un analista de datos experto y conciso, especializado en el análisis de "
                "archivos tabulares (Excel/CSV). Tu tarea es examinar los datos proporcionados, "
                "el contexto de las columnas seleccionadas y la pregunta del usuario. "
                "Genera un único párrafo de análisis en español, identificando tendencias, "
                "valores atípicos, o la respuesta más relevante a la pregunta. "
                "Sé profesional y directo. No uses encabezados ni Markdown para el formato."
            )
        }]
    }

    # 3. Construir el prompt del usuario (Pregunta y Datos)
    user_query = f"""
    Contexto del Análisis:
    - Columna(s) principal(es) del eje X (Categoría/Tiempo): '{eje_x}'
    - Columna(s) principal(es) del eje Y (Valor/Medida): '{eje_y}'
    - Tipo de Gráfico sugerido: '{tipo_grafico}'
    
    Pregunta Específica del Usuario: '{pregunta_nlp}'

    ---
    
    Datos a Analizar (Resumen y Muestra):
    {data_summary}
    
    Por favor, proporciona un insight analítico en español de un solo párrafo que responda a la pregunta del usuario basándote en los datos.
    """
    
    # 4. Construir el payload de la API
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": system_prompt,
    }

    # 5. Llamada a la API con reintento (Exponential Backoff)
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Lanza una excepción para errores 4xx/5xx

            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0]['text']
            
            return "La IA no pudo generar un insight claro. Intente con una pregunta diferente o más datos."

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                # st.warning(f"Error de API: {e}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"Error fatal al conectar con la IA después de {MAX_RETRIES} intentos: {e}")
                return "Error en la conexión con la IA. No se pudo generar el insight."
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar la respuesta de la IA: {e}")
            return "Error interno al procesar el insight de la IA."

    return "No se pudo generar el insight."


# ----------------------------------------------------
# 3. FUNCIÓN DE ACCIÓN: FILTRADO Y VISUALIZACIÓN
# ----------------------------------------------------
def seccion_analisis(df_original, pregunta_nlp):
    """
    Permite al usuario filtrar, seleccionar variables y visualizar datos.
    """
    
    # A. FILTROS BÁSICOS
    st.subheader("A. Filtrado de Datos")
    
    df = df_original.copy()
    columnas_disponibles = list(df.columns)
    
    with st.expander("Aplicar Filtros al DataFrame"):
        # Detectar columnas de texto (object o string)
        columnas_texto = [col for col in columnas_disponibles if df[col].dtype in ['object', 'string', 'category']]
        
        filtros_aplicados = False
        
        # Filtro por columnas de texto
        if columnas_texto:
            col_filtro_texto = st.selectbox("Columna para filtrar (Texto)", [''] + columnas_texto, index=0)
            if col_filtro_texto:
                valores_unicos = df[col_filtro_texto].unique()
                valores_seleccionados = st.multiselect(f"Seleccionar valores para {col_filtro_texto}", valores_unicos)
                if valores_seleccionados:
                    df = df[df[col_filtro_texto].isin(valores_seleccionados)]
                    filtros_aplicados = True
        
        # Filtro por columnas numéricas
        columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
        if columnas_numericas:
            col_filtro_num = st.selectbox("Columna para filtrar (Numérico)", [''] + columnas_numericas, index=0)
            if col_filtro_num:
                min_val = float(df[col_filtro_num].min())
                max_val = float(df[col_filtro_num].max())
                rango_seleccionado = st.slider(
                    f"Rango para {col_filtro_num}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=(min_val, max_val)
                )
                df = df[(df[col_filtro_num] >= rango_seleccionado[0]) & (df[col_filtro_num] <= rango_seleccionado[1])]
                filtros_aplicados = True

    if df.empty:
        st.warning("El conjunto de datos filtrado está vacío.")
        st.stop()

    # B. SELECCIÓN DE VARIABLES
    st.subheader("B. Configuración de Visualización")
    col1, col2 = st.columns(2)
    
    with col1:
        eje_x = st.selectbox("Eje X (Categoría/Agrupación/Tiempo)", columnas_disponibles)
    
    with col2:
        eje_y = st.selectbox("Eje Y (Valor/Medida)", [c for c in columnas_disponibles if c != eje_x])
    
    # C. TIPO DE GRÁFICO
    st.markdown("---")
    st.subheader("C. Selección de Gráfico y Agregación")
    
    tipos_grafico = [
        'Barras', 
        'Línea (Tendencia)', 
        'Dispersión (Scatter)', 
        'Histograma', 
        'Caja (Box Plot)', 
        'Circular (Pie Chart)'
    ]
    
    tipo_grafico = st.selectbox("Selecciona el tipo de gráfico", tipos_grafico)
    
    # Configuración de agregación para gráficos de Barras/Línea/Circular
    df_agregado = pd.DataFrame() # DataFrame para datos agregados

    if tipo_grafico in ['Barras', 'Línea (Tendencia)', 'Circular (Pie Chart)']:
        
        # CORRECCIÓN: Usar df[[eje_y]] para convertir la Series en un DataFrame de 1 columna
        # y así poder usar .columns.tolist() sin el AttributeError.
        try:
            columnas_numericas_y = df[[eje_y]].select_dtypes(include=['number']).columns.tolist()
        except KeyError:
            # Manejar el caso si la columna 'eje_y' no existe, aunque debería existir si se seleccionó arriba.
            # En la práctica, Streamlit asegura que 'eje_y' esté en 'df.columns' si no hay errores en la UI.
            columnas_numericas_y = [] 
        
        # Si el eje Y es numérico, ofrecemos agregación. Si no, contamos.
        if eje_y in columnas_numericas_y:
            metodos_agregacion = ['Suma', 'Promedio', 'Conteo']
            metodo_agregacion = st.selectbox("Método de Agregación", metodos_agregacion)
            
            y_col_name = f'{metodo_agregacion} de {eje_y}'
            
            if metodo_agregacion == 'Suma':
                df_agregado = df.groupby(eje_x, dropna=True)[eje_y].sum().reset_index(name=y_col_name)
            elif metodo_agregacion == 'Promedio':
                df_agregado = df.groupby(eje_x, dropna=True)[eje_y].mean().reset_index(name=y_col_name)
            elif metodo_agregacion == 'Conteo':
                df_agregado = df.groupby(eje_x, dropna=True).size().reset_index(name=y_col_name)
        else:
            # Conteo de ocurrencias si el eje Y no es numérico (e.g., Categorías)
            metodo_agregacion = 'Conteo'
            y_col_name = f'Conteo de {eje_y}'
            df_agregado = df.groupby(eje_x, dropna=True).size().reset_index(name=y_col_name)
            
        df_para_grafico = df_agregado
        x_col_name = eje_x

    else:
        # Para gráficos que usan datos brutos (Dispersión, Histograma, Caja)
        df_para_grafico = df
        df_agregado = pd.DataFrame() # Limpiar el agregado
    
    # D. VISUALIZACIÓN DEL GRÁFICO
    st.subheader("D. Gráfico Generado")
    
    if df_para_grafico.empty:
        st.warning("No hay datos para el gráfico después de la agregación o filtros.")
        return

    try:
        if tipo_grafico == 'Barras':
            fig = px.bar(df_para_grafico, x=x_col_name, y=y_col_name, title=f"Distribución: {y_col_name} por {eje_x}")

        elif tipo_grafico == 'Circular (Pie Chart)':
            fig = px.pie(
                df_para_grafico, 
                names=x_col_name, 
                values=y_col_name, 
                title=f"Distribución porcentual de {y_col_name} en {eje_x}"
            )

        elif tipo_grafico == 'Línea (Tendencia)':
            # Asegurar que el eje X pueda ser tratado como temporal o categórico ordenado
            fig = px.line(df_para_grafico, x=x_col_name, y=y_col_name, title=f"Tendencia: {metodo_agregacion} de {eje_y} a lo largo de {eje_x}")

        elif tipo_grafico == 'Dispersión (Scatter)':
            fig = px.scatter(df_para_grafico, x=eje_x, y=eje_y, title=f"Relación entre {eje_x} y {eje_y}", hover_data=columnas_disponibles)
            
        elif tipo_grafico == 'Histograma':
            # El histograma solo necesita un eje
            fig = px.histogram(df_para_grafico, x=eje_y, title=f"Distribución de {eje_y}")
            
        elif tipo_grafico == 'Caja (Box Plot)':
            fig = px.box(df_para_grafico, x=eje_x, y=eje_y, title=f"Distribución de {eje_y} por {eje_x}")
            
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al generar el gráfico. Asegúrate de que las columnas sean adecuadas para el tipo de gráfico: {e}")
        return # Detener la ejecución si el gráfico falla
    
    
    # ------------------------------------
    # E. INSIGHT GENERADO POR LENGUAJE NATURAL (LLM REAL)
    # ------------------------------------
    st.markdown("---")
    st.header("🧠 Insight Generado por NydIA (IA)")
    
    # Usar el DataFrame agregado para Linea/Barra/Circular, o el filtrado para el resto.
    df_insight = df_agregado if not df_agregado.empty else df
    
    if not df_insight.empty:
        with st.spinner("Analizando datos y generando insight con Gemini..."):
            # LLAMADA REAL A LA IA
            insight = generar_insight_con_gemini(df_insight, eje_x, eje_y, tipo_grafico, pregunta_nlp)
        
        # Mostrar el insight
        st.info(f"**Análisis Profundo de NydIA:**\n\n{insight}")
    else:
        st.info("No hay datos suficientes para generar un insight profundo.")


    st.markdown("---")
    st.caption(f"Filas originales consolidadas: {len(df_original)} | Filas analizadas después de filtros: {len(df)}")


# ----------------------------------------------------
# 4. EL BUCLE PRINCIPAL DEL AGENTE
# ----------------------------------------------------
def main():
    
    st.title("NydIA: Agente de Análisis de Datos con IA")
    st.markdown("Carga tus archivos y usa lenguaje natural para obtener *insights* impulsados por **Gemini**.")
    
    uploaded_files = st.file_uploader(
        "Carga tus archivos de Excel (.xls/.xlsx) o CSV (separado por comas/punto y coma):", 
        type=["xlsx", "xls", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        df_consolidado = consolidar_archivos(uploaded_files)
        
        if not df_consolidado.empty:
            
            st.success(f"Archivos cargados y consolidados: {len(df_consolidado)} filas y {len(df_consolidado.columns)} columnas.")
            
            # Mostrar el DataFrame (opcional)
            with st.expander("Ver Datos Consolidados (Primeras 10 filas)"):
                st.dataframe(df_consolidado.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Pregunta de Lenguaje Natural (NLQ - Natural Language Query)
            pregunta_nlp = st.text_input(
                "Pregunta a NydIA (e.g., ¿Cuál es el producto más vendido en el último trimestre?)",
                value="¿Qué tendencia muestra el promedio de ventas por región?" # Sugerencia
            )
            
            if st.button("🚀 Iniciar Análisis", type="primary"):
                seccion_analisis(df_consolidado, pregunta_nlp)

        else:
            st.warning("No se pudieron consolidar los datos de los archivos cargados.")
    else:
        st.info("Por favor, sube uno o más archivos de datos para comenzar el análisis.")

if __name__ == "__main__":
    main()