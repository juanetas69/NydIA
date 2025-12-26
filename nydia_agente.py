import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="NydIA: Multi-Análisis Pro")

# ----------------------------------------------------
# 1. FASE DE CARGA Y UNIÓN (CON CORRECCIÓN DE TIPOS)
# ----------------------------------------------------
def cargar_archivo_individual(file, sep):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension in ['xls', 'xlsx']:
            return pd.read_excel(io.BytesIO(file.getvalue()), engine='openpyxl')
        elif file_extension in ['csv', 'txt']:
            content = file.getvalue()
            # Intento de detección de encoding
            try:
                return pd.read_csv(io.BytesIO(content), encoding='utf-8', sep=sep)
            except:
                return pd.read_csv(io.BytesIO(content), encoding='latin-1', sep=sep)
    except Exception as e:
        st.error(f"Error en {file.name}: {e}")
        return None

def procesar_multiples_archivos(uploaded_files, modo, sep_choice, custom_sep):
    delimiters = {
        "Coma ( , )": ",", "Punto y Coma ( ; )": ";", 
        "Punto ( . )": ".", "Espacio ( )": " ", "Otro": custom_sep
    }
    selected_sep = delimiters.get(sep_choice, ",")
    
    dfs = {}
    for f in uploaded_files:
        df = cargar_archivo_individual(f, selected_sep)
        if df is not None:
            # Normalizar nombres de columnas: minúsculas y sin espacios
            df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
            dfs[f.name] = df

    if not dfs: return pd.DataFrame()

    if modo == "Apilar (Mismas columnas)":
        return pd.concat(dfs.values(), ignore_index=True)
    
    else: # MODO CRUCE (JOIN)
        st.sidebar.subheader("🔗 Configuración de Unión")
        all_cols = list(set().union(*(df.columns for df in dfs.values())))
        key_col = st.sidebar.selectbox("Columna clave para unir:", sorted(all_cols))
        
        main_df = None
        for name, df in dfs.items():
            if main_df is None:
                main_df = df
                # Normalizar clave a string para evitar ValueError de tipos mixtos
                if key_col in main_df.columns:
                    main_df[key_col] = main_df[key_col].astype(str).str.strip()
            else:
                if key_col in df.columns:
                    # Normalizar clave en el nuevo dataframe antes del merge
                    df_to_merge = df.copy()
                    df_to_merge[key_col] = df_to_merge[key_col].astype(str).str.strip()
                    
                    main_df = pd.merge(
                        main_df, 
                        df_to_merge, 
                        on=key_col, 
                        how='outer', 
                        suffixes=('', f'_{name}')
                    )
                else:
                    st.sidebar.warning(f"⚠️ '{key_col}' no encontrada en {name}. Se omitió de la unión.")
        return main_df

# ----------------------------------------------------
# 2. LÓGICA DE SUGERENCIA AUTOMÁTICA
# ----------------------------------------------------
def sugerir_grafico(df):
    """Analiza el dataframe y sugiere ejes y tipo de gráfico"""
    cols = df.columns.tolist()
    if not cols: return {"x": None, "y": None, "tipo": "Barras"}
    
    num_cols = [c for c in cols if is_numeric_dtype(df[c])]
    # Columnas categóricas con cardinalidad razonable
    cat_cols = [c for c in cols if not is_numeric_dtype(df[c]) and df[c].nunique() < 50]
    date_cols = [c for c in cols if is_datetime64_any_dtype(df[c])]

    # 1. Prioridad: Series Temporales
    if date_cols and num_cols:
        return {"x": date_cols[0], "y": num_cols[0], "tipo": "Líneas"}
    
    # 2. Prioridad: Comparación de Categorías
    if cat_cols and num_cols:
        if df[cat_cols[0]].nunique() <= 6:
            return {"x": cat_cols[0], "y": num_cols[0], "tipo": "Pie"}
        return {"x": cat_cols[0], "y": num_cols[0], "tipo": "Barras"}
    
    # 3. Prioridad: Correlación entre números
    if len(num_cols) >= 2:
        return {"x": num_cols[0], "y": num_cols[1], "tipo": "Dispersión"}
    
    # Fallback
    res_x = cat_cols[0] if cat_cols else cols[0]
    res_y = num_cols[0] if num_cols else cols[0]
    return {"x": res_x, "y": res_y, "tipo": "Barras"}

# ----------------------------------------------------
# 3. LIMPIEZA DE DATOS
# ----------------------------------------------------
def limpiar_datos_agresivo(df):
    if df.empty: return df
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Intento de conversión numérica para strings que parecen dinero o números
            sample = df_clean[col].dropna().astype(str).head(50).tolist()
            sample_str = "".join(sample)
            
            if any(char.isdigit() for char in sample_str):
                # Limpiar símbolos comunes
                temp_col = df_clean[col].astype(str).str.replace(r'[\$\€\s]', '', regex=True)
                # Manejo de formatos europeos/latinos (1.000,00)
                if temp_col.str.contains(r'\d\.\d{3},').any():
                    temp_col = temp_col.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                elif temp_col.str.contains(r'\d,\d').any() and not temp_col.str.contains(r'\d\.\d').any():
                    temp_col = temp_col.str.replace(',', '.', regex=False)
                
                numeric_conv = pd.to_numeric(temp_col, errors='coerce')
                if numeric_conv.notna().sum() > (len(df_clean) * 0.5):
                    df_clean[col] = numeric_conv
            
            # Intento de conversión a fecha
            try:
                date_conv = pd.to_datetime(df_clean[col], errors='coerce')
                if date_conv.notna().sum() > (len(df_clean) * 0.8):
                    df_clean[col] = date_conv
            except: pass
    return df_clean

# ----------------------------------------------------
# 4. INTERFAZ STREAMLIT
# ----------------------------------------------------
def main():
    st.title("🤖 NydIA: Agente de Análisis Multi-Archivo")
    
    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = None

    with st.sidebar:
        st.header("📂 Entrada")
        files = st.file_uploader("Sube archivos CSV/TXT/Excel", accept_multiple_files=True, type=['csv', 'txt', 'xlsx'])
        modo = st.radio("Estrategia de datos:", ["Apilar (Mismas columnas)", "Cruzar (Correlacionar archivos diferentes)"])
        
        st.markdown("---")
        st.subheader("⚙️ Configuración")
        sep_choice = st.selectbox("Delimitador (CSV/TXT):", ["Coma ( , )", "Punto y Coma ( ; )", "Punto ( . )", "Espacio ( )", "Otro"])
        custom_sep = st.text_input("Manual:") if sep_choice == "Otro" else ""

        if st.button("🚀 Procesar Datos"):
            if files:
                with st.spinner("Integrando y normalizando tipos..."):
                    raw = procesar_multiples_archivos(files, modo, sep_choice, custom_sep)
                    if not raw.empty:
                        st.session_state['df_raw'] = limpiar_datos_agresivo(raw)
                        st.success("¡Datos listos para analizar!")
            else:
                st.warning("Por favor, sube al menos un archivo.")

    if st.session_state['df_raw'] is not None:
        df_working = st.session_state['df_raw'].copy()
        
        # Filtros dinámicos en la barra lateral
        st.sidebar.header("🎯 Filtros Rápidos")
        cols_to_filter = st.sidebar.multiselect("Filtrar por columna:", df_working.columns.tolist())
        for col in cols_to_filter:
            if is_numeric_dtype(df_working[col]):
                min_v, max_v = float(df_working[col].min()), float(df_working[col].max())
                val = st.sidebar.slider(f"Rango {col}", min_v, max_v, (min_v, max_v))
                df_working = df_working[(df_working[col] >= val[0]) & (df_working[col] <= val[1])]
            elif is_datetime64_any_dtype(df_working[col]):
                dates = st.sidebar.date_input(f"Periodo {col}", [df_working[col].min(), df_working[col].max()])
                if len(dates) == 2:
                    df_working = df_working[(df_working[col].dt.date >= dates[0]) & (df_working[col].dt.date <= dates[1])]
            else:
                opts = sorted(df_working[col].dropna().unique().tolist())
                sel = st.sidebar.multiselect(f"Valores {col}", opts)
                if sel: df_working = df_working[df_working[col].isin(sel)]

        tab1, tab2, tab3 = st.tabs(["📊 Gráficos e IA", "🔍 Relaciones", "📄 Vista de Datos"])
        
        with tab3:
            st.info(f"Mostrando {len(df_working)} filas.")
            st.dataframe(df_working, use_container_width=True)

        with tab2:
            num_df = df_working.select_dtypes(include=[np.number])
            if num_df.shape[1] > 1:
                st.subheader("Mapa de Calor de Correlación")
                corr = num_df.corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Se requieren más columnas numéricas para calcular correlaciones.")

        with tab1:
            cols = df_working.columns.tolist()
            
            # PANEL DE CONTROL DE GRÁFICOS
            st.subheader("Configuración de Visualización")
            auto_mode = st.toggle("✨ Auto-Sugerir Gráfico (IA Sugerencia)", value=True)
            
            # Obtener sugerencia si el modo auto está activo
            sug = sugerir_grafico(df_working) if auto_mode else {"x": cols[0], "y": cols[0], "tipo": "Barras"}

            c1, c2, c3 = st.columns(3)
            with c1: 
                e_x = st.selectbox("Eje X (Categoría/Tiempo)", cols, 
                                 index=cols.index(sug["x"]) if sug["x"] in cols else 0)
            with c2: 
                e_y = st.selectbox("Eje Y (Valor Numérico)", cols, 
                                 index=cols.index(sug["y"]) if sug["y"] in cols else 0)
            with c3: 
                tipos = ["Barras", "Líneas", "Pie", "Dispersión", "Bigotes"]
                index_tipo = tipos.index(sug["tipo"]) if sug["tipo"] in tipos else 0
                tipo = st.selectbox("Formato del Gráfico", tipos, index=index_tipo)

            # RENDERIZADO
            try:
                if not is_numeric_dtype(df_working[e_y]) and tipo != "Pie":
                    st.warning(f"⚠️ La columna '{e_y}' no es numérica. El gráfico podría no ser preciso.")
                
                fig = None
                if tipo == "Barras":
                    df_p = df_working.groupby(e_x)[e_y].sum().reset_index()
                    fig = px.bar(df_p, x=e_x, y=e_y, color=e_x, title=f"Suma de {e_y} por {e_x}")
                elif tipo == "Líneas":
                    df_p = df_working.groupby(e_x)[e_y].sum().reset_index().sort_values(by=e_x)
                    fig = px.line(df_p, x=e_x, y=e_y, markers=True, title=f"Evolución de {e_y}")
                elif tipo == "Pie":
                    df_p = df_working.groupby(e_x)[e_y].sum().reset_index().sort_values(by=e_y, ascending=False).head(15)
                    fig = px.pie(df_p, names=e_x, values=e_y, title=f"Distribución de {e_y} (Top 15)")
                elif tipo == "Dispersión":
                    fig = px.scatter(df_working, x=e_x, y=e_y, color=e_x if df_working[e_x].nunique()<20 else None, title=f"Relación {e_x} vs {e_y}")
                elif tipo == "Bigotes":
                    fig = px.box(df_working, x=e_x, y=e_y, title=f"Análisis de Outliers: {e_y}")

                if fig:
                    fig.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error al generar visualización: {e}")

if __name__ == "__main__":
    main()