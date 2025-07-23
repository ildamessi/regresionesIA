import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

archivo_nuevo = pd.read_csv("datos_estudiantes_1000_nuevos_con_participacion.csv")

# Título
st.title("Predicción de Notas con Regresión Lineal Múltiple")

# Subir archivo CSV
st.sidebar.header("1. Cargar archivo CSV")
archivo = st.sidebar.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.subheader("Vista previa de los datos")
    st.write(df.head())

    # Visualización: Pairplot
    st.subheader("Relaciones entre variables")
    fig1 = sns.pairplot(df)
    st.pyplot(fig1)

    # Matriz de correlación
    st.subheader("Matriz de correlación")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Variables para el modelo
    X = df[['Horas', 'Asistencia', 'Tareas', 'Participacion']]
    y = df['Nota']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Evaluación
    st.subheader("Evaluación del Modelo")
    st.write("Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
    st.write("Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))

    # Gráfico de comparación
    st.subheader("Comparación: Notas Reales vs Predichas")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.5)
    ax3.plot([0, 10], [0, 10], 'r--')
    ax3.set_xlabel("Nota Real")
    ax3.set_ylabel("Nota Predicha")
    ax3.set_title("Comparación de Notas")
    st.pyplot(fig3)

    # Formulario de predicción
    st.sidebar.header("2. Predecir nueva nota")
    horas = st.sidebar.slider("Horas de Estudio", 0, 20, 10)
    asistencia = st.sidebar.slider("Asistencia (0-10)", 0, 10, 8)
    tareas = st.sidebar.slider("Tareas entregadas (0-10)", 0, 10, 6)
    participacion = st.sidebar.slider("Participación en clase (0-10)", 0, 10, 6)

    if st.sidebar.button("Predecir Nota"):
        nueva_pred = modelo.predict([[horas, asistencia, tareas, participacion]])
        st.sidebar.success(f"Nota estimada: {nueva_pred[0]:.2f}")