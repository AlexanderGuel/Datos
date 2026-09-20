"""
Matriz de correlacion para la hoja "IEDxAño" del archivo BD.

Uso:
    python matriz_correlacion_ied.py --archivo BD.xlsx
    python matriz_correlacion_ied.py --archivo BD.xlsx --metodo spearman --incluir-anio
    python matriz_correlacion_ied.py --archivo datos.csv

El archivo de entrada puede ser:
  - El .xlsx exportado desde Drive (Archivo > Descargar > Microsoft Excel),
    en cuyo caso se lee la pestaña indicada en --hoja (default "IEDxAño").
  - Un .csv con las mismas columnas (por si ya exportaste solo esa hoja).

Salidas (en --salida-dir, default "salida_correlacion"):
  - correlacion_ied.csv   -> matriz de correlacion completa
  - correlacion_ied.png   -> heatmap de la matriz
  - top_correlaciones_ied.csv -> variables mas correlacionadas con IED
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Columnas identificadoras: no son variables a correlacionar.
COLUMNAS_ID = ["Año", "Estado"]

# La hoja trae una columna separadora vacia sin nombre util.
COLUMNAS_A_DESCARTAR = ["."]

# Paleta divergente (rojo <-> gris neutro <-> azul) para correlaciones -1..1.
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "divergente_correlacion", ["#e34948", "#f0efec", "#2a78d6"]
)


def cargar_datos(archivo: Path, hoja: str) -> pd.DataFrame:
    if archivo.suffix.lower() == ".csv":
        return pd.read_csv(archivo)
    return pd.read_excel(archivo, sheet_name=hoja)


def limpiar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in COLUMNAS_A_DESCARTAR if c in df.columns])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # pandas renombra encabezados duplicados como "X", "X.1". En esta hoja
    # el segundo "IED Media" es en realidad el cociente IED Media / VAB
    # (mismo patron que "IED Media alta/VAB" e "IED Alta/VAB" al lado).
    if "IED Media.1" in df.columns:
        df = df.rename(columns={"IED Media.1": "IED Media/VAB"})

    return df


def construir_matriz_variables(df: pd.DataFrame, incluir_anio: bool) -> pd.DataFrame:
    columnas_id = COLUMNAS_ID if incluir_anio else [c for c in COLUMNAS_ID if c != "Año"]
    variables = df.drop(columns=[c for c in columnas_id if c in df.columns])
    variables = variables.apply(pd.to_numeric, errors="coerce")
    variables = variables.dropna(axis=1, how="all")
    return variables


def graficar_heatmap(corr: pd.DataFrame, salida: Path) -> None:
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(0.35 * n + 4, 0.35 * n + 4))

    im = ax.imshow(corr.values, cmap=CMAP_DIVERGENTE, vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)

    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Coeficiente de correlacion", color="#52514e")
    cbar.ax.tick_params(colors="#898781")

    ax.set_title("Matriz de correlacion - IEDxAño", color="#0b0b0b", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", required=True, type=Path, help="Ruta al .xlsx o .csv")
    parser.add_argument("--hoja", default="IEDxAño", help="Nombre de la hoja (solo .xlsx)")
    parser.add_argument(
        "--metodo", default="pearson", choices=["pearson", "spearman", "kendall"]
    )
    parser.add_argument(
        "--incluir-anio",
        action="store_true",
        help="Incluir 'Año' como variable numerica en la matriz",
    )
    parser.add_argument("--salida-dir", default=Path("salida_correlacion"), type=Path)
    args = parser.parse_args()

    args.salida_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(args.archivo, args.hoja)
    df = limpiar_columnas(df)
    variables = construir_matriz_variables(df, args.incluir_anio)

    corr = variables.corr(method=args.metodo)
    corr.to_csv(args.salida_dir / "correlacion_ied.csv")

    graficar_heatmap(corr, args.salida_dir / "correlacion_ied.png")

    if "IED" in corr.columns:
        top = (
            corr["IED"]
            .drop("IED")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top.to_csv(args.salida_dir / "top_correlaciones_ied.csv", header=["correlacion_con_IED"])
        print("Top 15 variables mas correlacionadas con IED:\n")
        print(top.head(15).round(3).to_string())

    print(f"\nListo. {len(variables.columns)} variables analizadas.")
    print(f"Resultados en: {args.salida_dir.resolve()}")


if __name__ == "__main__":
    main()
