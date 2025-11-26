import streamlit as st
import pandas as pd
import plotly.express as px
import io
# Usaremos 'io' para manejar el contenido binario de los archivos subidos.

# ----------------------------------------------------
# 1. FUNCIÓN DE PERCEPCIÓN Y CONSOLIDACIÓN (El 'Oído' del Agente)
# ----------------------------------------------------
# Tarea: Leer múltiples archivos Excel subidos por el usuario y unirlos en un solo DataFrame.

@st.cache_data # Streamlit "memoriza" el resultado si las entradas no cambian, ¡haciéndolo rápido!
def consolidar_archivos_excel(uploaded_files):
    """Procesa una lista de archivos subidos y devuelve un DataFrame consolidado."""
    
    # Si no hay archivos, no hay percepto.
    if not uploaded_files:
        return pd.DataFrame() 

    dataframes = []
    
    # Itera sobre cada archivo que el usuario ha subido
    for file in uploaded_files:
        try:
            # Lee el contenido del archivo subido. 
            # io.BytesIO(file.getvalue()) convierte el archivo de Streamlit en un objeto que pandas puede leer.
            df = pd.read_excel(io.BytesIO(file.getvalue()))
            dataframes.append(df)
        except Exception as e:
            # Muestra un mensaje de error si no puede leer alguno de los archivos.
            st.error(f"Error al leer el archivo {file.name}: {e}")
            
    # Combina todos los DataFrames apilándolos (uno debajo del otro)
    if dataframes:
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        return df_consolidado
    else:
        return pd.DataFrame()


# ----------------------------------------------------
# 2. FUNCIÓN DE ACCIÓN E INTERACCIÓN (El 'Cerebro' del Agente)
# ----------------------------------------------------
# Tarea: Interactuar con el usuario (pedir ejes) y generar la gráfica.

def interfaz_agente_analisis(df):
    """Crea la interfaz de Streamlit para la interacción y visualización."""
    
    st.title("📊 Agente de Análisis Libre y Compartible")
    st.markdown("---")
    
    if df.empty:
        st.warning("Por favor, sube uno o más archivos de Excel para que el agente pueda analizar los datos y generar gráficos.")
        return

    # Limpieza básica: El agente intenta convertir las columnas a tipos estándar
    df = df.infer_objects() 

    # ------------------------------------
    # A. INTERACCIÓN (Definición de Perceptos del Usuario)
    # ------------------------------------
    st.sidebar.header("⚙️ Configuración del Gráfico")
    
    # El agente identifica automáticamente las columnas numéricas y no numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    columnas_dimensiones = df.columns.tolist() # Se pueden usar todas las columnas como eje X

    if not columnas_numericas:
        st.error("El agente no encontró columnas con datos numéricos para graficar (Métrica).")
        return

    # El agente le pide al usuario que defina los ejes
    eje_x = st.sidebar.selectbox(
        "1. Selecciona la Dimensión (Eje X):", 
        columnas_dimensiones, 
        index=0 if columnas_dimensiones else None
    )
    eje_y = st.sidebar.selectbox(
        "2. Selecciona la Métrica (Eje Y):", 
        columnas_numericas,
        index=0 if columnas_numericas else None
    )
    tipo_grafico = st.sidebar.selectbox(
        "3. Selecciona el Tipo de Gráfico:", 
        ['Barras', 'Líneas', 'Dispersión (Scatter)']
    )

    # ------------------------------------
    # B. GENERACIÓN DE GRÁFICO (La Acción Final)
    # ------------------------------------
    
    st.subheader(f"Gráfico de **{tipo_grafico}** | {eje_y} vs {eje_x}")

    if tipo_grafico == 'Barras':
        # Para barras, agrupamos la dimensión para sumar o promediar la métrica
        df_agrupado = df.groupby(eje_x)[eje_y].sum().reset_index(name=f'Suma de {eje_y}')
        fig = px.bar(df_agrupado, x=eje_x, y=f'Suma de {eje_y}', 
                     title=f"Suma de {eje_y} por {eje_x}")
                     
    elif tipo_grafico == 'Líneas':
        fig = px.line(df, x=eje_x, y=eje_y, 
                      title=f"Tendencia de {eje_y} a lo largo de {eje_x}")
                      
    else: # Dispersión (Scatter)
        fig = px.scatter(df, x=eje_x, y=eje_y, 
                         title=f"Relación entre {eje_x} y {eje_y}")
        
    # Muestra el gráfico interactivo (característica de Plotly)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.caption(f"El agente ha consolidado {len(df)} filas de datos.")


# ----------------------------------------------------
# 3. EL BUCLE PRINCIPAL DEL AGENTE
# ----------------------------------------------------

def main():
    # PERCEPCIÓN (Entorno): Pide al usuario que suba los archivos de Excel
    uploaded_files = st.file_uploader(
        "Carga tus archivos de Excel (.xlsx o .xls) de la nube:", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True
    )
    
    # Lógica: Consolida los datos si hay archivos
    datos_consolidados = consolidar_archivos_excel(uploaded_files)
    
    # ACCIÓN: Lanza la interfaz de análisis
    interfaz_agente_analisis(datos_consolidados)

if __name__ == "__main__":
    main()