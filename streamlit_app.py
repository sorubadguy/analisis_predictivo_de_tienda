import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import io

# Configuración de página
st.set_page_config(page_title="Análisis de Ventas", layout="centered")

st.title("📊 Análisis Predictivo para Tienda Minorista")

# Cargar datos
ruta = 'Ventas.csv'

st.header("📁 Datos originales")
df_ventas = pd.read_csv(ruta)
st.dataframe(df_ventas)

# Info básica
st.subheader("🔍 Información del DataFrame")
buffer = []
st.subheader("🔍 Información del DataFrame")

buffer = io.StringIO()
df_ventas.info(buf=buffer)
st.text(buffer.getvalue())
st.text("".join(buffer))

st.subheader("📈 Estadísticas descriptivas")
st.dataframe(df_ventas.describe())

# Gráfico
st.subheader("📅 Ventas por Día de la Semana")
fig, ax = plt.subplots()
ax.scatter(df_ventas['DíaDeLaSemana'], df_ventas['Ventas'])
ax.set_title('Ventas por día de la semana')
ax.set_xlabel('Día de la semana')
ax.set_ylabel('Cantidad de ventas')
st.pyplot(fig)

# Modelado
st.subheader("🧠 Modelo de Regresión Logística")

dias_festivos = df_ventas.drop(['Promociones', 'Ventas'], axis=1)

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
    dias_festivos, df_ventas['Ventas'], train_size=0.9, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_entrena, y_entrena)

# Métrica
score = modelo.score(X_prueba, y_prueba)
st.metric(label="Precisión del modelo", value=f"{score:.2f}")

