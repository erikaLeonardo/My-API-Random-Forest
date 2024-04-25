from django.shortcuts import render
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier

def index(request):
    # Importar datos
    csv_file_path = os.path.join(os.path.dirname(__file__), 'TotalFeatures-ISCXFlowMeter.csv')
    df = pd.read_csv(csv_file_path)
    
    # Visualizar primeras filas y detalles del DataFrame
    df_head = df.head(10)
    df_info = df.info()
    df_value_counts = df['calss'].value_counts()

    # Copiar el conjunto de datos y transformar la variable de salida a numérica para calcular correlaciones
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]

    # Dividimos el conjunto de datos en entrenamiento, validación y prueba
    train_set, temp_set = train_test_split(X, test_size=0.3, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

    X_train, y_train = train_set.drop('calss', axis=1), train_set['calss']
    X_val, y_val = val_set.drop('calss', axis=1), val_set['calss']

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    MAX_DEPTH = 20

    # Instanciar y entrenar el modelo Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH, random_state=42)
    clf_rf.fit(X_train, y_train)

    # Predecir con el conjunto de datos de entrenamiento
    y_train_pred_rf = clf_rf.predict(X_train)

    # Predecir con el conjunto de datos de validación
    y_val_pred_rf = clf_rf.predict(X_val)

    # Calcular los puntajes F1 y F2 Score en el conjunto de datos de entrenamiento y validación
    f1_train_rf = f1_score(y_train, y_train_pred_rf, average='weighted')
    f1_val_rf = f1_score(y_val, y_val_pred_rf, average='weighted')

    f2_train_rf = fbeta_score(y_train, y_train_pred_rf, beta=2, average='weighted')
    f2_val_rf = fbeta_score(y_val, y_val_pred_rf, beta=2, average='weighted')

    return render(request, 'home/index.html', {
        'df_head': df_head,
        'df_info': df_info,
        'df_value_counts': df_value_counts,
        'f1_train_rf': f1_train_rf,
        'f1_val_rf': f1_val_rf,
        'f2_train_rf': f2_train_rf,
        'f2_val_rf': f2_val_rf
    })