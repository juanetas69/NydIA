import streamlit as st
import pandas as pd
import plotly.express as px
import io
import re
import time
import json
import base64

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="NydIA: Agente de Análisis Asistido por Gemini")

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
                # Lectura de CSV: Intentamos coma (,) y luego punto y coma (;), luego tab
                file_content = io.StringIO(file.getvalue().decode('utf-8', errors='ignore'))
                
                # Intentamos detectar el delimitador (comma, semicolon, or tab)
                try:
                    df = pd.read_csv(file_content, sep=',', on_bad_lines='skip')
                except Exception:
                    file_content.seek(0)
                    try:
                        df = pd.read_csv(file_content, sep=';', on_bad_lines='skip')
                    except Exception:
                        file_content.seek(0)
                        df = pd.read_csv(file_content, sep='\t', on_bad_lines='skip')
            else:
                st.warning(f"Formato no soportado para el archivo {file.name}. Se omitirá.")
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
# 2. FUNCIÓN DE LIMPIEZA Y PREPARACIÓN DE DATOS (Incluye Manejo de Fechas)
# ----------------------------------------------------
def limpiar_y_preparar_datos(df):
    """Limpia nombres de columnas y convierte tipos de datos, incluyendo fechas."""
    
    # 1. Limpieza de nombres de columnas
    nuevas_columnas = {}
    for col in df.columns:
        # Reemplazar caracteres especiales y espacios por guiones bajos
        limpio = re.sub(r'[^\w\s-]', '', str(col)).strip()
        limpio = re.sub(r'\s+', '_', limpio)
        limpio = limpio.lower()
        nuevas_columnas[col] = limpio
    df = df.rename(columns=nuevas_columnas)

    # 2. Conversión a tipos estándar y manejo de fechas
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        try:
            # Intentar convertir a numérico (útil para cadenas numéricas)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        except:
            # Si no es numérico, intentar convertir a datetime
            try:
                # Usar infer_datetime_format=True para mejor detección de formatos
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce', infer_datetime_format=True)
            except:
                # Si falla, intentar convertir a string para limpieza
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = df_cleaned[col].astype(str).str.strip().replace('nan', pd.NA).fillna(pd.NA)
    
    # Eliminar filas con todos los valores como NA/nulos después de la limpieza
    df_cleaned.dropna(how='all', inplace=True)
    
    return df_cleaned.infer_objects()

# ----------------------------------------------------
# 3. FUNCIÓN DE FILTRADO INTERACTIVO
# ----------------------------------------------------
def aplicar_filtros(df):
    """Aplica filtros interactivos al DataFrame y almacena el resultado en session_state."""
    
    df_filtrado = df.copy()
    
    st.sidebar.markdown("### 2. Filtros Dinámicos")
    
    # Identificar columnas para filtrado
    columnas_disponibles = df_filtrado.columns.tolist()
    columnas_filtrables = [col for col in columnas_disponibles if df_filtrado[col].nunique() < 50 and df_filtrado[col].dtype not in ['datetime64[ns]']]

    # Contenedor para los filtros
    with st.sidebar.expander("Añadir / Remover Filtros"):
        for col in columnas_filtrables:
            valores_unicos = sorted(df_filtrado[col].dropna().unique().tolist())
            
            # Crear un identificador de clave único para cada filtro
            key = f"filter_{col}"

            # Multiselect para aplicar el filtro
            seleccion = st.multiselect(
                f"Filtrar por: {col}",
                options=valores_unicos,
                default=[],
                key=key
            )
            
            if seleccion:
                # Filtrar el DataFrame
                df_filtrado = df_filtrado[df_filtrado[col].isin(seleccion)]

    # Filtros para columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    
    if columnas_numericas:
          with st.sidebar.expander("Filtros Numéricos (Rango)"):
            for col in columnas_numericas:
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Solo mostrar si hay un rango significativo
                if min_val != max_val:
                    rango = st.slider(
                        f"Rango para: {col}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val)),
                        key=f"slider_{col}"
                    )
                    df_filtrado = df_filtrado[(df_filtrado[col] >= rango[0]) & (df_filtrado[col] <= rango[1])]
    
    # Guardar el resultado del filtrado en el estado de la sesión
    st.session_state.df_filtrado = df_filtrado
    return df_filtrado

# ----------------------------------------------------
# 4. FUNCIÓN DE API LLAMADA (Para Análisis) - CORREGIDA Y ROBUSTA
# ----------------------------------------------------
def agente_analisis_llm(df, user_query):
    """Llama al modelo Gemini para análisis basado en un prompt del usuario y el resumen de datos."""
    
    # 1. Crear un resumen de datos para el modelo
    # Mostrar las primeras 5 filas y la estructura (dtypes)
    data_summary = f"Estructura del DataFrame (Columnas y Tipos):\n{df.dtypes.to_string()}\n\n"
    data_summary += f"Primeras 5 filas (para contexto de datos):\n{df.head().to_string()}"
    
    # 2. Construir el prompt para el modelo
    system_prompt = (
        "Eres un analista de datos experto y asistente de IA llamado NydIA. Tu tarea es analizar la 'consulta del usuario' "
        "en el contexto del 'resumen de datos' proporcionado (que incluye la estructura y una muestra de los datos). "
        "Genera una respuesta profesional, concisa y perspicaz en ESPAÑOL. "
        "Si la consulta es sobre análisis de negocios o tendencias, enfócate en los datos y los insights. "
        "Si la consulta es sobre cómo graficar, proporciona el mejor TIPO de gráfico y las COLUMNAS adecuadas (eje X, Y, Color, etc.) "
        "basándote en el resumen de datos para guiar al usuario a la sección 5."
    )
    
    user_query_full = f"Resumen de Datos:\n{data_summary}\n\nConsulta del Usuario: {user_query}\n\nRespuesta del Análisis:"
    
    # 3. Parámetros de la API
    # La apiKey se deja vacía para que el entorno de Canvas la inyecte automáticamente.
    apiKey = "" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query_full}]}],
        "tools": [{"google_search": {}}], # Útil para contexto general fuera de los datos
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # 4. Implementación de Backoff para la llamada API
    max_retries = 3
    base_delay = 1.0 # segundos

    for attempt in range(max_retries):
        try:
            # st.runtime.scriptrunner.fetch_wrapper is used to perform the fetch in the environment
            response = st.runtime.scriptrunner.fetch_wrapper(
                apiUrl, 
                method='POST', 
                headers=headers, 
                body=json.dumps(payload)
            )
            
            # Asumimos que response es un objeto con un método json()
            result = response.json()
            
            if response.status_code == 200 and 'candidates' in result and result['candidates']:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text
            elif response.status_code == 429:
                raise Exception("Tasa límite excedida (429)") # Forzamos el reintento
            else:
                st.warning(f"La API de Gemini devolvió un error (Estado: {response.status_code}).")
                return f"Error: No se pudo generar la respuesta de análisis. Código de estado: {response.status_code}"

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) # Retardo exponencial
                time.sleep(delay)
                continue # Intentar de nuevo
            else:
                return f"Error de comunicación con la API (después de {max_retries} intentos): {e}"
    
    return "Error desconocido en el proceso de análisis."

# ----------------------------------------------------
# 5. FUNCIÓN DE VISUALIZACIÓN INTERACTIVA (Gráficos)
# ----------------------------------------------------
def generar_grafico_interactivo(df_original, df):
    """Muestra un panel para seleccionar y generar gráficos interactivos con Plotly."""

    st.markdown("### 5. Generación de Gráficos Interactivos")
    
    # Identificar columnas por tipo
    columnas_disponibles = df.columns.tolist()
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    columnas_texto_fecha = [col for col in columnas_disponibles if col not in columnas_numericas]
    columnas_fecha = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    if not columnas_disponibles:
        st.info("No hay columnas disponibles para graficar. Asegúrate de que tus datos sean válidos.")
        return

    # Selecciones del usuario
    col1, col2, col3 = st.columns(3)

    with col1:
        tipo_grafico = st.selectbox(
            "Selecciona Tipo de Gráfico:",
            options=['Barras (Bar)', 'Línea (Line)', 'Dispersión (Scatter)', 'Histograma', 'Caja (Box Plot)', 'Circular (Pie)'],
            key='chart_type'
        )
    
    # Inicializar ejes a None
    eje_x = None
    eje_y = None

    # Lógica de selección de ejes basada en el tipo de gráfico
    if tipo_grafico in ['Barras (Bar)', 'Caja (Box Plot)', 'Circular (Pie)']:
        options_x = columnas_texto_fecha
        options_y = columnas_numericas
        
        with col2:
            eje_x = st.selectbox("Eje X / Etiquetas (Categoría):", options=options_x, index=0 if options_x else None, key='x_cat')
        with col3:
            eje_y = st.selectbox("Eje Y / Valores (Numérico):", options=[None] + options_y, index=1 if options_y else 0, key='y_num')

    elif tipo_grafico == 'Línea (Line)':
        options_x = columnas_fecha if columnas_fecha else columnas_texto_fecha
        options_y = columnas_numericas
        
        with col2:
            eje_x = st.selectbox("Eje X (Tiempo/Categoría):", options=options_x, index=0 if options_x else None, key='x_line')
        with col3:
            eje_y = st.selectbox("Eje Y (Valor Numérico):", options=[None] + options_y, index=1 if options_y else 0, key='y_line')
    
    elif tipo_grafico == 'Dispersión (Scatter)':
        options_num = columnas_numericas
        
        with col2:
            eje_x = st.selectbox("Eje X (Numérico):", options=options_num, index=0 if options_num else None, key='x_scatter')
        with col3:
            eje_y = st.selectbox("Eje Y (Numérico):", options=[None] + options_num, index=1 if options_num else 0, key='y_scatter')
        
    elif tipo_grafico == 'Histograma':
        options_num = columnas_numericas
        
        with col2:
            eje_y = st.selectbox("Columna (Numérica):", options=options_num, index=0 if options_num else None, key='y_hist')
        eje_x = None # No aplica

    
    # Botón de generación
    if st.button("Generar Gráfico", key='generate_chart_btn'):
        # Validación
        is_valid = True
        required_cols = []
        if tipo_grafico in ['Barras (Bar)', 'Línea (Line)', 'Caja (Box Plot)', 'Circular (Pie)']:
            if not eje_x:
                required_cols.append("Eje X")
            if not eje_y:
                required_cols.append("Eje Y")
        elif tipo_grafico == 'Dispersión (Scatter)':
            if not eje_x:
                required_cols.append("Eje X")
            if not eje_y:
                required_cols.append("Eje Y")
        elif tipo_grafico == 'Histograma':
            if not eje_y:
                required_cols.append("Columna")
        
        if required_cols:
            st.warning(f"Por favor, selecciona las columnas necesarias para el tipo de gráfico elegido: {', '.join(required_cols)}.")
            is_valid = False
            
        if is_valid and not df.empty:
            # FIX: Asegurar que se limpian los NaNs en las columnas usadas, para evitar el RangeError
            cols_to_clean = []
            if eje_x and eje_x != 'None': cols_to_clean.append(eje_x)
            if eje_y and eje_y != 'None': cols_to_clean.append(eje_y)

            if cols_to_clean:
                df_plot = df.dropna(subset=cols_to_clean).copy()
            else:
                df_plot = df.copy()

            if df_plot.empty:
                st.warning("El DataFrame queda vacío después de eliminar los valores faltantes (NaN) en las columnas seleccionadas.")
                return

            try:
                generar_plot(df_plot, tipo_grafico, eje_x, eje_y, columnas_disponibles)
            except Exception as e:
                st.error(f"Error al generar el gráfico: {e}")
        elif df.empty:
            st.warning("No hay datos filtrados disponibles para graficar.")

def generar_plot(df, tipo_grafico, eje_x, eje_y, columnas_disponibles):
    """Función de Plotly para generar el gráfico."""
    
    # Lógica de agregación para gráficos de Barras/Línea/Circular
    y_col_name = eje_y if eje_y else "conteo_de_filas"
    df_agregado = df.copy()

    if tipo_grafico in ['Barras (Bar)', 'Línea (Line)', 'Circular (Pie)'] and eje_x:
        st.sidebar.markdown("##### Opciones de Agregación")
        metodo_agregacion = st.sidebar.selectbox(
             "Método de Agregación:",
             options=['Suma', 'Promedio', 'Conteo'],
             key='agg_method'
         )

        if metodo_agregacion == 'Conteo':
            df_agregado = df.groupby(eje_x).size().reset_index(name='Conteo')
            y_col_name = 'Conteo'
        elif eje_y is None or eje_y not in df.select_dtypes(include=['number']).columns.tolist():
             # Forzamos conteo si Y no es numérica y no se especificó un método de conteo
             df_agregado = df.groupby(eje_x).size().reset_index(name='Conteo')
             y_col_name = 'Conteo'
        elif metodo_agregacion == 'Suma':
            df_agregado = df.groupby(eje_x)[eje_y].sum().reset_index(name=f"Suma_de_{eje_y}")
            y_col_name = f"Suma_de_{eje_y}"
        elif metodo_agregacion == 'Promedio':
            df_agregado = df.groupby(eje_x)[eje_y].mean().reset_index(name=f"Promedio_de_{eje_y}")
            y_col_name = f"Promedio_de_{eje_y}"
        
        titulo_base = f"{metodo_agregacion} de {eje_y if eje_y else 'Filas'} por {eje_x}"

        if tipo_grafico == 'Barras (Bar)':
            fig = px.bar(df_agregado, x=eje_x, y=y_col_name, title=f"Barras: {titulo_base}")

        elif tipo_grafico == 'Línea (Line)':
            # Asegurarse de que el eje X esté ordenado si es una columna de fecha
            if df_agregado[eje_x].dtype == 'datetime64[ns]':
                df_agregado = df_agregado.sort_values(eje_x)
            fig = px.line(df_agregado, x=eje_x, y=y_col_name, title=f"Línea: {titulo_base}")

        elif tipo_grafico == 'Circular (Pie)':
            # Para el gráfico circular, la columna de etiquetas se llama 'names' y la de valores 'values'
            fig = px.pie(df_agregado, names=eje_x, values=y_col_name, title=f"Circular: Distribución porcentual de {y_col_name} por {eje_x}")
            
    # Gráficos sin agregación (usan el DF limpio directamente)
    elif tipo_grafico == 'Dispersión (Scatter)':
        fig = px.scatter(df, x=eje_x, y=eje_y, title=f"Dispersión: Relación entre {eje_x} y {eje_y}", hover_data=columnas_disponibles)
        
    elif tipo_grafico == 'Histograma':
        fig = px.histogram(df, x=eje_y, title=f"Histograma: Distribución de {eje_y}")
        
    elif tipo_grafico == 'Caja (Box Plot)':
        fig = px.box(df, x=eje_x, y=eje_y, title=f"Caja: Distribución de {eje_y} por {eje_x}")
        
    st.plotly_chart(fig, use_container_width=True)

    
# ----------------------------------------------------
# 6. FUNCIÓN DE DESCARGA DE DATAFRAME
# ----------------------------------------------------
def descargar_dataframe(df, filename="datos_filtrados.csv"):
    """Genera un botón de descarga para el DataFrame."""
    
    # Convertir el DataFrame a CSV con delimitador de punto y coma (más compatible con Excel en español)
    csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    
    # Crear el enlace de descarga
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="st-emotion-cache-nahz7x e1nzilvr5">Descargar Datos Filtrados ({len(df)} filas)</a>'
    st.markdown(href, unsafe_allow_html=True)


# ----------------------------------------------------
# 7. EL BUCLE PRINCIPAL DEL AGENTE
# ----------------------------------------------------
def main():
    
    # Inicialización del estado de sesión
    if 'df_original' not in st.session_state:
        st.session_state.df_original = pd.DataFrame()
    if 'df_filtrado' not in st.session_state:
        st.session_state.df_filtrado = pd.DataFrame()
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = ""
        
    # --- Columna Lateral para Carga y Filtros ---
    with st.sidebar:
        st.header("1. Carga de Datos")
        uploaded_files = st.file_uploader(
            "Carga tus archivos de datos (.csv, .xls/.xlsx)",
            type=['csv', 'xls', 'xlsx'],
            accept_multiple_files=True
        )
        
        # Lógica de carga y consolidación
        if uploaded_files and (st.session_state.df_original.empty or st.button("Recargar Archivos", key='reload_btn')):
            with st.spinner('Consolidando y limpiando datos...'):
                df_cargado = consolidar_archivos(uploaded_files)
                if not df_cargado.empty:
                    st.session_state.df_original = limpiar_y_preparar_datos(df_cargado)
                    st.session_state.df_filtrado = st.session_state.df_original.copy()
                    st.session_state.llm_response = "" # Limpiar respuesta de IA
                    st.success("Archivos consolidados y listos para el análisis.")
                else:
                    st.session_state.df_original = pd.DataFrame()
                    st.error("No se pudieron cargar datos válidos.")

    
    df_original = st.session_state.df_original
    
    if df_original.empty:
        st.info("Por favor, carga uno o más archivos para comenzar el análisis de NydIA.")
        return

    # --- Aplicación de Filtros (si el DF original existe) ---
    df_actualizado = aplicar_filtros(df_original)
    
    # --- Contenido Principal de la Aplicación ---
    st.title("NydIA 🧠: Agente de Análisis de Datos Asistido por IA")
    
    col_viz, col_data_info = st.columns([3, 1])

    with col_data_info:
        st.markdown("### 3. Resumen de Datos")
        st.metric("Filas Originales", len(df_original))
        st.metric("Filas Filtradas", len(df_actualizado))
        st.metric("Columnas", len(df_actualizado.columns))
        
        # Botón de descarga
        st.markdown("---")
        descargar_dataframe(df_actualizado)
        
        st.markdown("---")
        st.markdown("#### Estructura de Datos (DTypes)")
        st.dataframe(df_actualizado.dtypes.astype(str).reset_index().rename(columns={'index': 'Columna', 0: 'Tipo'}), 
                     hide_index=True, use_container_width=True)


    with col_viz:
        st.markdown("### 4. Asistente de Análisis (Gemini)")
        user_query = st.text_area(
            "Escribe tu pregunta o solicitud de análisis (ej. 'Analiza la tendencia de las ventas por mes', '¿Cuál es el mejor gráfico para correlacionar precio y cantidad?'):",
            key='llm_query',
            height=100
        )

        if st.button("Ejecutar Análisis", key='run_llm'):
            if user_query:
                with st.spinner("Analizando con Gemini (esto puede tardar unos segundos)..."):
                    # Usar una muestra si el DF filtrado es muy grande (ej. > 1000 filas)
                    df_to_analyze = df_actualizado.sample(min(1000, len(df_actualizado))) if len(df_actualizado) > 1000 else df_actualizado
                    
                    # Llamar al agente
                    respuesta = agente_analisis_llm(df_to_analyze, user_query)
                    st.session_state.llm_response = respuesta
            else:
                st.warning("Por favor, ingresa una consulta para ejecutar el análisis.")
        
        if st.session_state.llm_response:
             st.markdown("#### 💬 Respuesta del Agente NydIA:")
             st.markdown(st.session_state.llm_response)
        else:
             st.info("La respuesta de la IA aparecerá aquí después de ejecutar el análisis.")

        st.markdown("---")
        
        # --- Sección de Visualización ---
        generar_grafico_interactivo(df_original, df_actualizado)


if __name__ == '__main__':
    main()