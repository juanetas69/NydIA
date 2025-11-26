import streamlit as st
import pandas as pd
import plotly.express as px
import io

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Agente de Análisis Libre")

# ----------------------------------------------------
# 1. FUNCIÓN DE PERCEPCIÓN Y CONSOLIDACIÓN (El 'Oído' del Agente)
# ----------------------------------------------------
@st.cache_data
def consolidar_archivos_excel(uploaded_files):
    """Procesa una lista de archivos subidos y devuelve un DataFrame consolidado."""
    
    if not uploaded_files:
        return pd.DataFrame() 

    dataframes = []
    
    for file in uploaded_files:
        try:
            # Lee el contenido binario del archivo subido.
            df = pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error al leer el archivo {file.name}: {e}")
            
    if dataframes:
        # Combina todos los DataFrames apilándolos (uno debajo del otro)
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # Intentar inferir tipos para mejor manejo
        df_consolidado = df_consolidado.infer_objects() 
        return df_consolidado
    else:
        return pd.DataFrame()


# ----------------------------------------------------
# 2. FUNCIÓN DE ACCIÓN E INTERACCIÓN (El 'Cerebro' del Agente)
# ----------------------------------------------------
def interfaz_agente_analisis(df_original):
    """Crea la interfaz de Streamlit para la interacción, filtrado y visualización."""
    
    st.title("📊 Agente de Análisis Libre y Compartible (Avanzado)")
    st.markdown("Este agente consolida tus archivos de Excel, permite aplicar filtros dinámicos y genera gráficos interactivos con Plotly.", help="Desarrollado con Software Libre: Python, Pandas y Streamlit.")
    st.markdown("---")
    
    if df_original.empty:
        st.warning("Por favor, sube uno o más archivos de Excel para que el agente pueda analizar los datos.")
        return

    # Creamos una copia del DataFrame para aplicar los filtros
    df = df_original.copy()
    
    # ------------------------------------
    # A. FILTROS DINÁMICOS
    # ------------------------------------
    st.sidebar.header("🔍 1. Aplicar Filtros")
    
    # Filtros de Texto (Categorías)
    st.sidebar.subheader("Filtros por Categoría:")
    
    # Solo mostrar filtros para columnas de texto que no tengan demasiados valores únicos
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # Si tiene demasiados valores únicos (ej. más de 50), lo ignoramos para no saturar la interfaz
        if df[col].nunique() <= 50:
            opciones_filtro = ['TODOS'] + sorted(df[col].dropna().unique().tolist())
            seleccion = st.sidebar.selectbox(f"Filtrar por **{col}**:", opciones_filtro)
            
            if seleccion != 'TODOS':
                df = df[df[col] == seleccion]
            
    
    # Filtro de Rango Numérico
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro por Rango:")
    
    columnas_numericas = df_original.select_dtypes(include=['number']).columns.tolist()
    
    if columnas_numericas:
        col_num_a_filtrar = st.sidebar.selectbox("Columna a Filtrar por Rango:", ['Seleccionar'] + columnas_numericas)
        
        if col_num_a_filtrar != 'Seleccionar':
            min_val = float(df_original[col_num_a_filtrar].min())
            max_val = float(df_original[col_num_a_filtrar].max())
            
            rango_seleccionado = st.sidebar.slider(
                f"Rango de {col_num_a_filtrar}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                # El paso se ajusta dinámicamente para ser el 1% del rango total
                step=max(0.01, (max_val - min_val) / 100)
            )
            # Aplicamos el filtro al DataFrame
            df = df[
                (df[col_num_a_filtrar] >= rango_seleccionado[0]) & 
                (df[col_num_a_filtrar] <= rango_seleccionado[1])
            ]
    
    # Si después de los filtros el DataFrame está vacío, notificamos al usuario
    if df.empty:
        st.error("No hay datos para graficar después de aplicar los filtros. Intenta suavizar los filtros.")
        st.markdown(f"Filas originales: {len(df_original)} | Filas filtradas: 0")
        return

    # ------------------------------------
    # B. CONFIGURACIÓN DEL GRÁFICO
    # ------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("📈 2. Configuración de Gráfico")
    
    columnas_disponibles = df.columns.tolist() 
    columnas_numericas_filtradas = df.select_dtypes(include=['number']).columns.tolist()

    if not columnas_numericas_filtradas:
        st.error("La selección actual no contiene columnas numéricas para la Métrica (Eje Y).")
        return

    # Selecciones del usuario
    eje_x = st.sidebar.selectbox(
        "Dimensión (Eje X):", 
        columnas_disponibles, 
        index=0 if columnas_disponibles else None
    )
    eje_y = st.sidebar.selectbox(
        "Métrica (Eje Y):", 
        columnas_numericas_filtradas,
        index=0 if columnas_numericas_filtradas else None
    )

    tipo_grafico = st.sidebar.selectbox(
        "Tipo de Gráfico:", 
        ['Barras', 'Líneas', 'Dispersión (Scatter)', 'Histograma', 'Caja (Box Plot)']
    )

    # Opciones de Agregación, solo para gráficos que lo requieren
    metodo_agregacion = 'Ninguna'
    if tipo_grafico in ['Barras', 'Líneas']:
        metodo_agregacion = st.sidebar.selectbox(
            "Método de Agregación:", 
            ['Suma', 'Promedio', 'Conteo']
        )
    
    
    # ------------------------------------
    # C. GENERACIÓN DEL GRÁFICO
    # ------------------------------------
    
    st.subheader(f"Análisis | Tipo: **{tipo_grafico}** | Filas analizadas: {len(df)}")

    try:
        if tipo_grafico in ['Barras', 'Líneas']:
            
            # 1. Agregación de datos
            if metodo_agregacion == 'Suma':
                df_agregado = df.groupby(eje_x)[eje_y].sum().reset_index(name=f'Suma de {eje_y}')
            elif metodo_agregacion == 'Promedio':
                df_agregado = df.groupby(eje_x)[eje_y].mean().reset_index(name=f'Promedio de {eje_y}')
            else: # Conteo
                df_agregado = df.groupby(eje_x).size().reset_index(name='Conteo de Elementos')
            
            # 2. Creación del gráfico
            y_col_name = df_agregado.columns[-1] 
            
            if tipo_grafico == 'Barras':
                fig = px.bar(df_agregado, x=eje_x, y=y_col_name, title=f"{metodo_agregacion} de {eje_y} por {eje_x}")
            else:
                fig = px.line(df_agregado, x=eje_x, y=y_col_name, title=f"Tendencia: {metodo_agregacion} de {eje_y} a lo largo de {eje_x}")

        elif tipo_grafico == 'Dispersión (Scatter)':
            fig = px.scatter(df, x=eje_x, y=eje_y, title=f"Relación entre {eje_x} y {eje_y}", hover_data=columnas_disponibles)
            
        elif tipo_grafico == 'Histograma':
            # Eje X puede ser cualquier columna, pero se mide el conteo o distribución de una métrica
            fig = px.histogram(df, x=eje_y, title=f"Distribución de {eje_y}", histfunc=metodo_agregacion.lower() if metodo_agregacion != 'Ninguna' else None)
            
        elif tipo_grafico == 'Caja (Box Plot)':
            # Muestra estadísticas clave (mediana, cuartiles) de la métrica por dimensión
            fig = px.box(df, x=eje_x, y=eje_y, title=f"Distribución de {eje_y} por {eje_x}")
            
        # Muestra el gráfico interactivo
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al generar el gráfico. Asegúrate de que las columnas seleccionadas sean adecuadas para el tipo de gráfico: {e}")
    
    st.markdown("---")
    st.caption(f"Filas originales consolidadas: {len(df_original)} | Filas analizadas después de filtros: {len(df)}")


# ----------------------------------------------------
# 3. EL BUCLE PRINCIPAL DEL AGENTE
# ----------------------------------------------------
def main():
    
    # PERCEPCIÓN (Entorno): Pide al usuario que suba los archivos de Excel
    uploaded_files = st.file_uploader(
        "Carga tus archivos de Excel (.xlsx o .xls):", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True
    )
    
    # Lógica: Consolida los datos si hay archivos
    datos_consolidados = consolidar_archivos_excel(uploaded_files)
    
    # ACCIÓN: Lanza la interfaz de análisis
    interfaz_agente_analisis(datos_consolidados)

if __name__ == "__main__":
    main()