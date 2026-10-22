"""
Segmentación de clientes con K-Means – Despliegue con Streamlit
Ejecutar:  streamlit run app.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")

RANDOM_STATE = 42
FEATURES = ["Ingreso_Anual_k", "Puntaje_Gasto"]

ESTRATEGIAS = {
    "Clientes de alto valor": "Retener y fidelizar: programa VIP, preventas exclusivas, atención personalizada y venta cruzada de productos premium.",
    "Clientes promedio": "Aumentar frecuencia y ticket: combos, cupones por recurrencia, puntos acumulables y up-selling gradual.",
    "Clientes promedio – gastadores jóvenes": "Redes sociales e influenciadores, tendencias, descuentos estudiantiles, pago a cuotas y gamificación.",
    "Clientes con potencial": "Activar: entender por qué no compran, ofertas de alta calidad, experiencias exclusivas, asesoría personalizada y conveniencia.",
    "Clientes de bajo valor": "Eficiencia: comunicación masiva de bajo costo, ofertas en básicos, temporadas de descuento y marcas propias.",
}
COLORES = {
    "Clientes de alto valor": "#2ca02c",
    "Clientes promedio": "#1f77b4",
    "Clientes promedio – gastadores jóvenes": "#9467bd",
    "Clientes con potencial": "#ff7f0e",
    "Clientes de bajo valor": "#d62728",
}


# ------------------------------------------------------------------ datos y modelo
@st.cache_data
def cargar_datos():
    ruta = "Mall_Customers.xls"
    try:
        df = pd.read_excel(ruta)
    except Exception:          # el .xls es realmente un CSV
        df = pd.read_csv(ruta)
    return df.rename(columns={
        "Gender": "Genero", "Genre": "Genero", "Age": "Edad",
        "Annual Income (k$)": "Ingreso_Anual_k",
        "Spending Score (1-100)": "Puntaje_Gasto",
    })


def asignar_perfil(ingreso, gasto, ing_med, gas_med, banda=12):
    if abs(ingreso - ing_med) <= banda and abs(gasto - gas_med) <= banda:
        return "Clientes promedio"
    if ingreso >= ing_med and gasto >= gas_med:
        return "Clientes de alto valor"
    if ingreso >= ing_med:
        return "Clientes con potencial"
    if gasto >= gas_med:
        return "Clientes promedio – gastadores jóvenes"
    return "Clientes de bajo valor"


@st.cache_resource
def preparar_modelo(df, k):
    X = df[FEATURES].values
    # Usa el modelo guardado desde el cuaderno si existe y coincide con K
    if k == 5 and os.path.exists("modelo_kmeans.pkl"):
        art = joblib.load("modelo_kmeans.pkl")
        return art["scaler"], art["modelo"], art["perfiles"]
    scaler = StandardScaler().fit(X)
    modelo = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaler.transform(X))
    cent = scaler.inverse_transform(modelo.cluster_centers_)
    ing_med, gas_med = df["Ingreso_Anual_k"].median(), df["Puntaje_Gasto"].median()
    perfiles = {i: asignar_perfil(c[0], c[1], ing_med, gas_med) for i, c in enumerate(cent)}
    return scaler, modelo, perfiles


@st.cache_data
def curvas(df, k_max=10):
    Xs = StandardScaler().fit_transform(df[FEATURES].values)
    inercias, siluetas = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(Xs, km.labels_) if k > 1 else np.nan)
    return inercias, siluetas


df = cargar_datos()

# ------------------------------------------------------------------ barra lateral
st.sidebar.title("⚙️ Configuración")
k = st.sidebar.slider("Número de clusters (K)", 2, 10, 5,
                      help="K = 5 es el valor óptimo según el codo y la silueta.")
scaler, modelo, perfiles = preparar_modelo(df, k)
Xs = scaler.transform(df[FEATURES].values)
df["Cluster"] = modelo.predict(Xs)
df["Perfil"] = df["Cluster"].map(perfiles)
centroides = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=FEATURES)
centroides["Perfil"] = centroides.index.map(perfiles)

st.sidebar.metric("Silueta del modelo", f"{silhouette_score(Xs, df['Cluster']):.3f}")
st.sidebar.metric("Inercia (WCSS)", f"{modelo.inertia_:.1f}")
st.sidebar.caption(f"Dataset: Mall Customers · {len(df)} clientes")

st.title("🛍️ Segmentación de clientes con K-Means")
tab1, tab2, tab3 = st.tabs(["📉 Método del codo", "🎯 Clusters", "🧑 Clasificar cliente"])

# ------------------------------------------------------------------ 1. codo
with tab1:
    inercias, siluetas = curvas(df)
    ks = list(range(1, 11))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inercias, "o-", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set(title="Método del codo", xlabel="K", ylabel="Inercia (WCSS)", xticks=ks)
        ax.legend()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks[1:], siluetas[1:], "s-", color="seagreen", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set(title="Coeficiente de silueta", xlabel="K", ylabel="Silueta", xticks=ks[1:])
        ax.legend()
        st.pyplot(fig)
    tabla = pd.DataFrame({"K": ks, "Inercia": np.round(inercias, 2), "Silueta": np.round(siluetas, 3)})
    st.dataframe(tabla.set_index("K").T)
    st.info("El codo aparece en **K = 5**, y es también el K con la silueta más alta (≈ 0.555). "
            "Por eso se eligió K = 5.")

# ------------------------------------------------------------------ 2. clusters
with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="Ingreso_Anual_k", y="Puntaje_Gasto", hue="Perfil",
                    palette=COLORES, s=70, alpha=.85, ax=ax)
    ax.scatter(centroides["Ingreso_Anual_k"], centroides["Puntaje_Gasto"], s=300, c="black",
               marker="X", label="Centroides")
    ax.set(title=f"Clusters de clientes (K = {k})", xlabel="Ingreso anual (k$)",
           ylabel="Puntaje de gasto (1-100)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

    resumen = df.groupby(["Cluster", "Perfil"]).agg(
        Clientes=("CustomerID", "count"),
        Edad_media=("Edad", "mean"),
        Ingreso_medio=("Ingreso_Anual_k", "mean"),
        Gasto_medio=("Puntaje_Gasto", "mean"),
    ).round(1).reset_index()
    st.subheader("Resumen por cluster")
    st.dataframe(resumen, hide_index=True)

    with st.expander("Ver datos segmentados"):
        st.dataframe(df, hide_index=True)
        st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                           "clientes_segmentados.csv", "text/csv")

# ------------------------------------------------------------------ 3. nuevo cliente
with tab3:
    st.subheader("Ingresar un nuevo cliente")
    with st.form("nuevo_cliente"):
        a, b = st.columns(2)
        with a:
            ingreso = st.number_input("Ingreso anual (k$)", 0.0, 300.0, 60.0, step=1.0)
            gasto = st.slider("Puntaje de gasto (1-100)", 1, 100, 50)
        with b:
            edad = st.number_input("Edad (informativa)", 18, 100, 35)
            genero = st.selectbox("Género (informativo)", ["Female", "Male"])
        enviar = st.form_submit_button("Clasificar cliente", type="primary")

    if enviar:
        nuevo = pd.DataFrame([[ingreso, gasto]], columns=FEATURES)
        cluster = int(modelo.predict(scaler.transform(nuevo.values))[0])
        perfil = perfiles[cluster]

        st.success(f"El cliente pertenece al **Cluster {cluster}: {perfil}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cluster", cluster)
        m2.metric("Ingreso medio del cluster", f"{centroides.loc[cluster, 'Ingreso_Anual_k']:.1f} k$")
        m3.metric("Gasto medio del cluster", f"{centroides.loc[cluster, 'Puntaje_Gasto']:.1f}")
        st.markdown(f"**Estrategia recomendada:** {ESTRATEGIAS[perfil]}")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Ingreso_Anual_k", y="Puntaje_Gasto", hue="Perfil",
                        palette=COLORES, s=60, alpha=.35, ax=ax)
        ax.scatter(centroides["Ingreso_Anual_k"], centroides["Puntaje_Gasto"], s=250, c="black", marker="X")
        ax.scatter(ingreso, gasto, s=500, c="gold", marker="*", edgecolors="black",
                   linewidths=1.5, label="Nuevo cliente", zorder=5)
        ax.set(title="Ubicación del nuevo cliente", xlabel="Ingreso anual (k$)",
               ylabel="Puntaje de gasto (1-100)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig)
