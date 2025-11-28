import streamlit as st
import pandas as pd
import plotly.express as px
import io
import re

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="NydIA: Análisis Multi-Formato con Gráfico Pie")

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
                    df = pd.read_csv(file_content, sep=',', encoding='utf-8', on_bad_lines='skip')
                except pd.errors.ParserError:
                    # Intento 2: Punto y coma como delimitador
                    file_content.seek(0)
                    df = pd.read_csv(file_content, sep=';', encoding='utf-8', on_bad_lines='skip')
            else:
                st.warning(f"Tipo de archivo no soportado: {file.name}")
                continue
                
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error al leer el archivo {file.name}: {e}")
            
    if dataframes:
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        # Limpiamos nombres de columnas (eliminar espacios al inicio/final)
        df_consolidado.columns = [col.strip() for col in df_consolidado.columns]
        # Intentar inferir objetos para asegurar la correcta lectura de tipos
        df_consolidado = df_consolidado.infer_objects() 
        return df_consolidado
    else:
        return pd.DataFrame()

# ----------------------------------------------------
# 2. FUNCIÓN DE FILTRADO (Implementación básica)
# ----------------------------------------------------
def aplicar_filtros(df, filtros_str):
    """Aplica filtros básicos basados en una cadena de texto."""
    if not filtros_str:
        return df
    
    df_filtrado = df.copy()
    
    # Patrón básico para capturar la lógica (Ejemplo: 'Columna == "Valor"')
    # Este es un ejemplo muy simple y debería ser mejorado para un agente real.
    try:
        # Utilizamos eval para aplicar filtros dinámicos, lo cual puede ser peligroso 
        # pero es simple para este ejemplo. En un entorno real se recomienda un parser más seguro.
        # Aseguramos que la sintaxis de Pandas sea correcta
        df_filtrado = df_filtrado.query(filtros_str, engine='python')
    except Exception as e:
        st.warning(f"Error en la sintaxis de filtro '{filtros_str}': {e}. Se devuelve el DataFrame sin filtrar.")
        return df
        
    return df_filtrado

# ----------------------------------------------------
# 3. FUNCIÓN DE VISUALIZACIÓN (Gráficos)
# ----------------------------------------------------
def generar_grafico(df_original, df, tipo_grafico, eje_x, eje_y, metodo_agregacion, columnas_disponibles):
    """Genera y muestra un gráfico de Plotly basado en los parámetros seleccionados."""
    
    if df.empty:
        st.warning("El DataFrame está vacío. No se puede generar el gráfico.")
        return

    # ----------------------------------------------------------------------
    # FIX para RangeError: Eliminar NaNs en las columnas clave antes de graficar
    # Esto previene el error 'values={[NaN,NaN]} needs to be sorted'
    # ----------------------------------------------------------------------
    cols_to_check = []
    if eje_x and eje_x != 'Ninguno':
        cols_to_check.append(eje_x)
    if eje_y:
        cols_to_check.append(eje_y)
        
    df_limpio = df.copy()
    
    if cols_to_check:
        # Solo eliminamos NaNs si hay columnas seleccionadas para el gráfico
        df_limpio = df.dropna(subset=cols_to_check)
        
    if df_limpio.empty:
        st.warning(f"Advertencia: Después de aplicar filtros y eliminar valores faltantes (NaN) en las columnas del gráfico ({', '.join(cols_to_check)}), el conjunto de datos está vacío. No se puede generar el gráfico.")
        st.markdown("---")
        st.caption(f"Filas originales consolidadas: {len(df_original)} | Filas analizadas después de filtros y limpieza: {len(df_limpio)}")
        return
        
    df = df_limpio # Usamos el DataFrame limpio para generar el gráfico.
    
    # ----------------------------------------------------------------------
    
    try:
        fig = None
        
        if tipo_grafico == 'Tarta (Pie)':
            # Agrupar por eje_x (categoría) y sumar o contar eje_y (valor)
            if eje_x and eje_y:
                # Si eje_y es numérico, lo agregamos (sumamos), si no, simplemente contamos la frecuencia de eje_x
                if pd.api.types.is_numeric_dtype(df[eje_y]):
                    df_agregado = df.groupby(eje_x)[eje_y].sum().reset_index()
                    names_col = eje_x
                    values_col = eje_y
                else:
                    df_agregado = df[eje_x].value_counts().reset_index()
                    df_agregado.columns = [eje_x, 'Conteo']
                    names_col = eje_x
                    values_col = 'Conteo'
                
                fig = px.pie(df_agregado, names=names_col, values=values_col, title=f"Distribución: {eje_x} vs. {values_col}")
                
        elif tipo_grafico in ['Línea', 'Barras']:
            y_col_name = f'{metodo_agregacion} de {eje_y}'
            
            # Agregación: Agrupa por eje_x y aplica el método (sum, mean, count) a eje_y
            df_agregado = df.groupby(eje_x)[eje_y].agg(metodo_agregacion).reset_index()
            df_agregado.columns = [eje_x, y_col_name]
            
            if tipo_grafico == 'Línea':
                fig = px.line(df_agregado, x=eje_x, y=y_col_name, title=f"Tendencia: {metodo_agregacion} de {eje_y} a lo largo de {eje_x}")
            else: # Barras
                fig = px.bar(df_agregado, x=eje_x, y=y_col_name, title=f"Comparación: {metodo_agregacion} de {eje_y} por {eje_x}")

        elif tipo_grafico == 'Dispersión (Scatter)':
            fig = px.scatter(df, x=eje_x, y=eje_y, title=f"Relación entre {eje_x} y {eje_y}", hover_data=columnas_disponibles)
            
        elif tipo_grafico == 'Histograma':
            fig = px.histogram(df, x=eje_y, title=f"Distribución de {eje_y}")
            
        elif tipo_grafico == 'Caja (Box Plot)':
            fig = px.box(df, x=eje_x, y=eje_y, title=f"Distribución de {eje_y} por {eje_x}")
            
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selección de gráfico o ejes inválida. Asegúrate de que las columnas sean adecuadas.")

    except Exception as e:
        # Capturamos otros errores de Plotly o Pandas
        st.error(f"Ocurrió un error al generar el gráfico. Asegúrate de que las columnas sean adecuadas para el tipo de gráfico: {e}")
    
    st.markdown("---")
    st.caption(f"Filas originales consolidadas: {len(df_original)} | Filas analizadas después de filtros y limpieza: {len(df)}")


# ----------------------------------------------------
# 4. EL BUCLE PRINCIPAL DEL AGENTE
# ----------------------------------------------------
def main():
    
    uploaded_files = st.file_uploader(
        "Carga tus archivos de Excel (.xls/.xlsx) o CSV para analizar:",
        type=['xls', 'xlsx', 'csv'],
        accept_multiple_files=True
    )
    
    df_original = consolidar_archivos(uploaded_files)

    if df_original.empty:
        st.info("Sube uno o más archivos para empezar el análisis.")
        return

    st.subheader(f"Datos Consolidados ({len(df_original)} filas)")
    columnas_disponibles = list(df_original.columns)

    # Inicializar estado para filtros
    if 'filtros' not in st.session_state:
        st.session_state.filtros = ''
        
    st.markdown("---")

    # Sidebar para controles
    with st.sidebar:
        st.header("Controles de Análisis")
        
        # 1. Filtro
        st.subheader("1. Filtrado de Datos (Query de Pandas)")
        st.session_state.filtros = st.text_area(
            "Escribe tu condición de filtro (Ej: 'Columna > 10' o 'Categoría == \"A\"')",
            value=st.session_state.filtros,
            key="filtro_input"
        )
        
        # 2. Tipo de Gráfico
        st.subheader("2. Configuración de Gráfico")
        
        tipos_grafico = ['Tarta (Pie)', 'Línea', 'Barras', 'Dispersión (Scatter)', 'Histograma', 'Caja (Box Plot)']
        tipo_grafico = st.selectbox("Selecciona Tipo de Gráfico:", tipos_grafico)

        eje_x = st.selectbox("Eje X (Categoría o Tiempo):", ['Ninguno'] + columnas_disponibles)
        eje_y = st.selectbox("Eje Y (Valor Numérico/Distribución):", columnas_disponibles)
        
        # Configuración de agregación solo para Línea/Barras
        metodo_agregacion = 'sum'
        if tipo_grafico in ['Línea', 'Barras']:
            st.caption("Método de Agregación para Eje Y:")
            metodo_agregacion = st.radio(
                "Agregación:", 
                ('sum', 'mean', 'count', 'min', 'max'),
                index=0,
                horizontal=True
            )
            
    # Aplicar Filtros
    df_filtrado = aplicar_filtros(df_original, st.session_state.filtros)

    # ----------------------------------------------------
    # 5. ZONA DE RESULTADOS
    # ----------------------------------------------------

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("DataFrame Filtrado")
        if df_filtrado.empty:
            st.warning("El filtro aplicado resultó en un DataFrame vacío.")
        else:
            st.dataframe(df_filtrado.head(10), use_container_width=True)
            st.markdown(f"**Filas después de filtros:** {len(df_filtrado)}")

    with col2:
        st.subheader(f"Visualización: {tipo_grafico}")
        generar_grafico(df_original, df_filtrado, tipo_grafico, eje_x, eje_y, metodo_agregacion, columnas_disponibles)

if __name__ == "__main__":
    main()