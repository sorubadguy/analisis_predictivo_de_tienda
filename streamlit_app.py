import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
buffer = io.StringIO()
df_ventas.info(buf=buffer)
st.text(buffer.getvalue())

st.subheader("📈 Estadísticas descriptivas")
st.dataframe(df_ventas.describe())

# Gráfico de dispersión
st.subheader("📅 Ventas por Día de la Semana")
fig, ax = plt.subplots()
ax.scatter(df_ventas['DíaDeLaSemana'], df_ventas['Ventas'])
ax.set_title('Ventas por día de la semana')
ax.set_xlabel('Día de la semana')
ax.set_ylabel('Cantidad de ventas')
st.pyplot(fig)

# Modelado
st.subheader("🧠 Modelo de Regresión Lineal")

# Preparar datos
dias_festivos = df_ventas.drop(['Promociones', 'Ventas'], axis=1)
dias_festivos = pd.get_dummies(dias_festivos)

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
    dias_festivos, df_ventas['Ventas'], train_size=0.9, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_entrena, y_entrena)

# Evaluación del modelo
score = modelo.score(X_prueba, y_prueba)

st.subheader("🎯 R² del modelo (ajuste)")
st.metric(label="Coeficiente de determinación", value=f"{score:.2f}")

# Predicciones
st.subheader("🔮 Predicciones")
y_pred = modelo.predict(X_prueba)

resultados = pd.DataFrame({
    'Real': y_prueba.values,
    'Predicción': y_pred.round(2)
})
st.dataframe(resultados.head())

# Gráfico real vs predicción
st.subheader("📊 Comparación: Valores Reales vs Predichos")
fig2, ax2 = plt.subplots()
ax2.scatter(y_prueba, y_pred, alpha=0.6)
ax2.plot([y_prueba.min(), y_prueba.max()], [y_prueba.min(), y_prueba.max()], 'r--')
ax2.set_xlabel("Ventas reales")
ax2.set_ylabel("Ventas predichas")
ax2.set_title("Ventas: Real vs Predicho")
st.pyplot(fig2)
