import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # ¡Nueva librería para escalar!

# 1. Cargar los datos
print("Cargando datos...")
df = pd.read_csv('online_shoppers_intention.csv')

# --- NUEVO: 2. Convertir Textos a Números (Encoding) ---
print("Transformando variables de texto a números...")
# Convertimos True/False a 1 y 0
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Convertimos los meses y tipos de visitante a columnas binarias (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

# 3. Separar las variables de entrada (X) y el objetivo (y)
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# 4. Dividir en Entrenamiento (80%) y Prueba/Unseen Data (20%)
# Agregamos "stratify=y" para que el 20% de prueba mantenga la misma proporción de compras
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- NUEVO: 5. Escalar los datos (Standardization) ---
print("Escalando los datos numéricos...")
scaler = StandardScaler()

# OJO AQUÍ: "Ajustamos" la regla matemática solo con los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Y aplicamos esa misma regla a los datos ocultos (unseen data) para no hacer trampa
X_test_scaled = scaler.transform(X_test)

print(f"\n¡Preprocesamiento completado!")
print(f"Ahora tenemos {X_train_scaled.shape[1]} columnas puramente numéricas listas para el algoritmo.")


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- NUEVO: 6. Inicializar y Entrenar el Algoritmo ---
print("\nIniciando el entrenamiento de la Red Neuronal (Perceptrón Multicapa)...")
# Configuramos la red neuronal con 2 capas ocultas (por ejemplo, de 50 y 25 neuronas)
# max_iter=500 le da tiempo suficiente para aprender sin detenerse antes de tiempo
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)

# El modelo "estudia" usando solo los datos de entrenamiento
mlp_model.fit(X_train_scaled, y_train)
print("¡Entrenamiento finalizado!")

# --- NUEVO: 7. Evaluar el modelo con los DATOS OCULTOS (Unseen Data) ---
print("\nEvaluando el modelo con el 20% de datos nunca vistos (Unseen Data)...")
y_pred = mlp_model.predict(X_test_scaled)

# 8. Mostrar los resultados
print("\n=== RESULTADOS DE LA PREDICCIÓN ===")
print(f"Precisión General (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Matriz de Confusión:")
# Muestra los verdaderos positivos, falsos positivos, etc.
print(confusion_matrix(y_test, y_pred))

print("\nReporte Detallado de Clasificación:")
print(classification_report(y_test, y_pred))