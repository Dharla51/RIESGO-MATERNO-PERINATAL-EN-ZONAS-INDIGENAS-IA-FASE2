
"""
SISTEMA DE ALERTA TEMPRANA DE RIESGO MATERNO-PERINATAL
FASE 2 - INFERENCIA DE MODELOS DE INTELIGENCIA ARTIFICIAL

Modelos:
- Regresion Logistica
- Support Vector Machine (SVM)
- Red Neuronal Multicapa (MLP)

Tareas:
1. Clasificacion del nivel de riesgo
2. Prediccion binaria de desenlaces adversos

IMPORTANTE:
Prototipo academico basado en datos sinteticos.
No esta validado para uso clinico.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras


# ==========================================
# 1. CONFIGURACION DE RUTAS
# ==========================================

RAIZ = Path(__file__).resolve().parent

MODELOS = RAIZ / "Modelos"
RESULTADOS = RAIZ / "Resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

CONFIGURACION = RAIZ / "configuracion.json"

UMBRALES = {
    "riesgo_mlp_umbral_alto": 0.25,
    "desenlace_logistica_umbral": 0.30,
    "desenlace_mlp_umbral": 0.50
}

if CONFIGURACION.exists():
    with open(CONFIGURACION, "r", encoding="utf-8") as f:
        configuracion = json.load(f)

    for clave in UMBRALES:
        if clave in configuracion:
            UMBRALES[clave] = float(configuracion[clave])


# ==========================================
# 2. FUNCIONES DE CARGA
# ==========================================

def verificar_archivo(ruta):
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ruta}\n"
            "Verifica que exportaste los modelos "
            "desde Google Colab."
        )


def cargar_joblib(nombre):
    ruta = MODELOS / nombre
    verificar_archivo(ruta)
    return joblib.load(ruta)


def cargar_keras(nombre):
    ruta = MODELOS / nombre
    verificar_archivo(ruta)

    return keras.models.load_model(
        ruta,
        compile=False
    )


def cargar_modelos():
    print("\nCargando modelos de inteligencia artificial...")

    modelos = {
        # Modelos de nivel de riesgo
        "riesgo_mlp": cargar_keras(
            "riesgo_mlp.keras"
        ),
        "riesgo_logistica": cargar_joblib(
            "riesgo_logistica.joblib"
        ),
        "riesgo_svm": cargar_joblib(
            "riesgo_svm.joblib"
        ),

        # Modelos de desenlace adverso
        "desenlace_mlp": cargar_keras(
            "desenlace_mlp.keras"
        ),
        "desenlace_logistica": cargar_joblib(
            "desenlace_logistica.joblib"
        ),
        "desenlace_svm": cargar_joblib(
            "desenlace_svm.joblib"
        ),

        # Preprocesadores y codificadores
        "riesgo_componentes": cargar_joblib(
            "riesgo_componentes.joblib"
        ),
        "desenlace_componentes": cargar_joblib(
            "desenlace_componentes.joblib"
        )
    }

    print("Modelos cargados correctamente.")
    return modelos


# ==========================================
# 3. CARGAR Y VALIDAR DATASET
# ==========================================

def cargar_dataset(ruta_csv):

    ruta = Path(ruta_csv)
    verificar_archivo(ruta)

    df = pd.read_csv(
        ruta,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("El archivo CSV esta vacio.")

    print("\nDataset cargado:")
    print("Registros:", df.shape[0])
    print("Columnas:", df.shape[1])

    return df


def preparar_entradas(df, componentes):

    columnas = componentes["columnas"]

    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(faltantes)
        )

    # Conservar exactamente el orden del entrenamiento
    X = df[columnas].copy()

    # Asegurar formato numerico cuando corresponda
    for columna in X.columns:
        if columna in [
            "edad_materna",
            "semanas_gestacion_control",
            "numero_controles_prenatales",
            "distancia_centro_salud_km",
            "tiempo_traslado_horas",
            "hemoglobina_g_dl",
            "presion_sistolica",
            "presion_diastolica",
            "imc",
            "antecedente_preeclampsia",
            "antecedente_diabetes_gestacional",
            "parto_institucional_previo"
        ]:
            X[columna] = pd.to_numeric(
                X[columna],
                errors="coerce"
            )

    return X


# ==========================================
# 4. PREDICCION DE NIVEL DE RIESGO
# ==========================================

def predecir_riesgo(X, modelos):

    print("\nEvaluando nivel de riesgo...")

    componentes = modelos["riesgo_componentes"]

    procesador = componentes["preprocesador"]
    codificador = componentes["encoder"]

    # Aplicar el mismo preprocesamiento de Colab
    X_procesado = procesador.transform(X)

    X_procesado = np.asarray(
        X_procesado,
        dtype=np.float32
    )

    # Red neuronal
    probabilidades = modelos["riesgo_mlp"].predict(
        X_procesado,
        verbose=0
    )

    clases = np.asarray(codificador.classes_)

    indices_originales = np.argmax(
        probabilidades,
        axis=1
    )

    riesgo_mlp = clases[indices_originales]

    # Umbral experimental para riesgo alto
    indice_alto = np.where(
        clases == "Alto"
    )[0][0]

    indices_otros = np.where(
        clases != "Alto"
    )[0]

    mejor_otra_clase = indices_otros[
        np.argmax(
            probabilidades[:, indices_otros],
            axis=1
        )
    ]

    umbral = UMBRALES["riesgo_mlp_umbral_alto"]

    indices_ajustados = np.where(
        probabilidades[:, indice_alto] >= umbral,
        indice_alto,
        mejor_otra_clase
    )

    riesgo_ajustado = clases[indices_ajustados]

    # Modelos tradicionales
    pred_logistica = modelos[
        "riesgo_logistica"
    ].predict(X)

    pred_svm = modelos[
        "riesgo_svm"
    ].predict(X)

    return pd.DataFrame({
        "riesgo_logistica": pred_logistica,
        "riesgo_svm": pred_svm,
        "riesgo_mlp": riesgo_mlp,
        "riesgo_mlp_umbral": riesgo_ajustado,
        "puntuacion_riesgo_alto_mlp": np.round(
            probabilidades[:, indice_alto], 4
        )
    })


# ==========================================
# 5. PREDICCION DE DESENLACE ADVERSO
# ==========================================

def predecir_desenlace(X, modelos):

    print("\nEvaluando desenlaces adversos...")

    componentes = modelos["desenlace_componentes"]

    procesador = componentes["preprocesador"]

    X_procesado = procesador.transform(X)

    X_procesado = np.asarray(
        X_procesado,
        dtype=np.float32
    )

    # Regresion Logistica
    modelo_lr = modelos["desenlace_logistica"]

    clases_lr = np.asarray(modelo_lr.classes_)
    indice_positivo = np.where(
        clases_lr == 1
    )[0][0]

    prob_lr = modelo_lr.predict_proba(
        X
    )[:, indice_positivo]

    pred_lr = (
        prob_lr >= UMBRALES["desenlace_logistica_umbral"]
    ).astype(int)

    # SVM
    pred_svm = modelos[
        "desenlace_svm"
    ].predict(X)

    # Red neuronal
    prob_mlp = modelos["desenlace_mlp"].predict(
        X_procesado,
        verbose=0
    ).ravel()

    pred_mlp = (
        prob_mlp >= UMBRALES["desenlace_mlp_umbral"]
    ).astype(int)

    etiquetas = {
        0: "Normal",
        1: "Posible desenlace adverso"
    }

    return pd.DataFrame({
        "desenlace_logistica": pd.Series(
            pred_lr
        ).map(etiquetas),

        "desenlace_svm": pd.Series(
            pred_svm
        ).map(etiquetas),

        "desenlace_mlp": pd.Series(
            pred_mlp
        ).map(etiquetas),

        "puntuacion_adverso_logistica": np.round(
            prob_lr, 4
        ),

        "puntuacion_adverso_mlp": np.round(
            prob_mlp, 4
        )
    })


# ==========================================
# 6. EJECUTAR TODAS LAS PREDICCIONES
# ==========================================

def ejecutar_predicciones(ruta_csv, ruta_salida=None):

    print("=" * 55)
    print("SISTEMA DE ALERTA MATERNO-PERINATAL")
    print("PROTOTIPO EXPERIMENTAL - FASE 2")
    print("=" * 55)

    datos = cargar_dataset(ruta_csv)

    modelos = cargar_modelos()

    # Preparar variables para ambas tareas
    X_riesgo = preparar_entradas(
        datos,
        modelos["riesgo_componentes"]
    )

    X_desenlace = preparar_entradas(
        datos,
        modelos["desenlace_componentes"]
    )

    # Ejecutar clasificacion de riesgo
    resultado_riesgo = predecir_riesgo(
        X_riesgo,
        modelos
    )

    # Ejecutar prediccion de desenlace
    resultado_desenlace = predecir_desenlace(
        X_desenlace,
        modelos
    )

    # Consolidar resultados
    resultados = pd.concat(
        [
            resultado_riesgo.reset_index(drop=True),
            resultado_desenlace.reset_index(drop=True)
        ],
        axis=1
    )

    # Numero de registro para la demostracion
    resultados.insert(
        0,
        "registro",
        np.arange(1, len(resultados) + 1)
    )

    # Mostrar predicciones
    print("\n===== RESULTADOS =====")

    print(
        resultados.head(10).to_string(index=False)
    )

    print("\n===== RESUMEN DE RIESGO =====")

    print(
        resultados["riesgo_mlp"].value_counts()
    )

    print("\n===== RESUMEN DE DESENLACE =====")

    print(
        resultados["desenlace_logistica"].value_counts()
    )

    # Guardar CSV
    if ruta_salida is None:
        ruta_salida = RESULTADOS / "predicciones_demo.csv"
    else:
        ruta_salida = Path(ruta_salida)

    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPredicciones guardadas en:")
    print(ruta_salida.resolve())

    print("\nADVERTENCIA:")
    print(
        "Estos modelos fueron entrenados con datos "
        "sinteticos y NO poseen validacion clinica."
    )

    return resultados


# ==========================================
# 7. PUNTO DE ENTRADA
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predicciones experimentales de riesgo "
            "materno-perinatal mediante IA."
        )
    )

    parser.add_argument(
        "csv",
        help="Ruta del CSV de entrada"
    )

    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta opcional del CSV de resultados"
    )

    args = parser.parse_args()

    ejecutar_predicciones(
        args.csv,
        args.salida
    )


if __name__ == "__main__":
    main()
