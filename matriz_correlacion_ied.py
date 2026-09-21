"""
Matriz de correlacion y significancia estadistica para la hoja "IEDxAño"
del archivo BD.

Uso:
    python matriz_correlacion_ied.py --archivo BD.xlsx
    python matriz_correlacion_ied.py --archivo BD.xlsx --metodo spearman
    python matriz_correlacion_ied.py --archivo BD.xlsx --variables-foco "IED,PIB (M MXN precios del 2018)"

El archivo de entrada puede ser:
  - El .xlsx exportado desde Drive (Archivo > Descargar > Microsoft Excel),
    en cuyo caso se lee la pestaña indicada en --hoja (default "IEDxAño").
  - Un .csv con las mismas columnas (por si ya exportaste solo esa hoja).

Salidas (en --salida-dir, default "salida_correlacion"):
  - correlacion_ied.csv          -> matriz de correlacion completa
  - pvalores_ied.csv             -> matriz de p-valores (misma forma)
  - correlacion_ied.png          -> heatmap general; celdas no significativas
                                     (p >= 0.05) se muestran atenuadas
  - detalle_<variable>.csv       -> por cada variable en --variables-foco:
                                     correlacion, p-valor, n y significancia
                                     contra el resto de variables
  - detalle_<variable>.png       -> barras horizontales de las relaciones
                                     mas fuertes de esa variable
  - regresion_<objetivo>.csv     -> coeficiente estandarizado, error estandar,
                                     t y p-valor de una regresion multiple que
                                     explica --objetivo-regresion (default PIB),
                                     tras excluir variables casi-identicas al
                                     objetivo y podar colinealidad (VIF)
  - regresion_<objetivo>.png     -> grafico de coeficientes con IC 95%
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm_api
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Columnas identificadoras: no son variables a correlacionar.
COLUMNAS_ID = ["Año", "Estado"]

# La hoja trae una columna separadora vacia sin nombre util.
COLUMNAS_A_DESCARTAR = ["."]

# Paleta divergente (rojo <-> gris neutro <-> azul) para correlaciones -1..1.
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "divergente_correlacion", ["#e34948", "#f0efec", "#2a78d6"]
)
AZUL, ROJO = "#2a78d6", "#e34948"
ALPHA_NO_SIGNIFICATIVO = 0.15
NIVEL_SIGNIFICANCIA = 0.05


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


def _correlacion_par(x: pd.Series, y: pd.Series, metodo: str):
    valido = x.notna() & y.notna()
    n = int(valido.sum())
    if n < 3:
        return np.nan, np.nan, n
    funcion = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[metodo]
    r, p = funcion(x[valido], y[valido])
    return float(r), float(p), n


def calcular_correlacion_y_significancia(variables: pd.DataFrame, metodo: str):
    """Devuelve (corr, p_valor, n_obs), tres DataFrames cuadrados alineados."""
    cols = variables.columns
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    pval = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    nobs = pd.DataFrame(np.zeros((n, n), dtype=int), index=cols, columns=cols)

    for i in range(n):
        for j in range(i + 1, n):
            r, p, n_ij = _correlacion_par(variables.iloc[:, i], variables.iloc[:, j], metodo)
            corr.iloc[i, j] = corr.iloc[j, i] = r
            pval.iloc[i, j] = pval.iloc[j, i] = p
            nobs.iloc[i, j] = nobs.iloc[j, i] = n_ij
        nobs.iloc[i, i] = int(variables.iloc[:, i].notna().sum())

    return corr, pval, nobs


def estrellas(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < NIVEL_SIGNIFICANCIA:
        return "*"
    return ""


def resolver_columna(nombre: str, columnas) -> str:
    if nombre in columnas:
        return nombre
    coincidencias = [c for c in columnas if nombre.lower() in c.lower()]
    if len(coincidencias) == 1:
        return coincidencias[0]
    opciones = ", ".join(coincidencias[:5]) if coincidencias else "(sin coincidencias)"
    raise SystemExit(
        f"No se encontro la variable '{nombre}' de forma exacta. Coincidencias parciales: {opciones}"
    )


def graficar_heatmap_general(corr: pd.DataFrame, pval: pd.DataFrame, salida: Path) -> None:
    n = len(corr.columns)
    rgba = CMAP_DIVERGENTE((corr.values + 1) / 2)
    no_significativo = (pval.values >= NIVEL_SIGNIFICANCIA) & ~np.eye(n, dtype=bool)
    rgba[..., 3] = np.where(no_significativo, ALPHA_NO_SIGNIFICATIVO, 1.0)

    fig, ax = plt.subplots(figsize=(0.35 * n + 4, 0.35 * n + 4))
    ax.imshow(rgba)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)

    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for spine in ax.spines.values():
        spine.set_visible(False)

    mapeo_color = plt.cm.ScalarMappable(cmap=CMAP_DIVERGENTE, norm=plt.Normalize(-1, 1))
    cbar = fig.colorbar(mapeo_color, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Coeficiente de correlacion", color="#52514e")
    cbar.ax.tick_params(colors="#898781")

    ax.set_title(
        f"Matriz de correlacion - IEDxAño\nCeldas atenuadas: no significativas al {int(NIVEL_SIGNIFICANCIA*100)}%",
        color="#0b0b0b",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def tabla_detalle_variable(
    variable: str, corr: pd.DataFrame, pval: pd.DataFrame, nobs: pd.DataFrame
) -> pd.DataFrame:
    tabla = pd.DataFrame(
        {
            "correlacion": corr[variable],
            "p_valor": pval[variable],
            "n_obs": nobs[variable],
        }
    ).drop(index=variable)
    tabla["significativo_5pct"] = tabla["p_valor"] < NIVEL_SIGNIFICANCIA
    tabla["nivel"] = tabla["p_valor"].apply(estrellas)
    tabla = tabla.reindex(tabla["correlacion"].abs().sort_values(ascending=False).index)
    tabla.index.name = "variable"
    return tabla


def graficar_barras_significancia(
    tabla: pd.DataFrame,
    columna_valor: str,
    titulo: str,
    etiqueta_x: str,
    salida: Path,
    top_n: int,
    columna_error: str | None = None,
) -> None:
    """Barras horizontales de una magnitud con signo (r o coeficiente),
    atenuando lo no significativo. Reutilizada por el detalle por variable
    (correlacion) y por la regresion multiple (coeficiente estandarizado)."""
    datos = tabla.head(top_n).iloc[::-1]
    etiquetas = [textwrap.fill(str(v), 38) for v in datos.index]
    colores = [AZUL if v >= 0 else ROJO for v in datos[columna_valor]]
    alphas = [1.0 if sig else ALPHA_NO_SIGNIFICATIVO + 0.25 for sig in datos["significativo_5pct"]]
    errores = 1.96 * datos[columna_error] if columna_error else None

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(datos) + 2), constrained_layout=True)
    barras = ax.barh(
        etiquetas,
        datos[columna_valor],
        xerr=errores,
        color=colores,
        error_kw={"ecolor": "#898781", "elinewidth": 1, "capsize": 3},
    )
    for barra, alpha in zip(barras, alphas):
        barra.set_alpha(alpha)

    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", labelsize=8, colors="#0b0b0b")
    ax.tick_params(axis="x", colors="#898781")

    x0, x1 = ax.get_xlim()
    margen = max(abs(x0), abs(x1), 0.05)
    ax.set_xlim(x0 - margen * 0.18, x1 + margen * 0.18)
    desplazamiento = margen * 0.04
    extremos = errores if errores is not None else pd.Series(0.0, index=datos.index)
    for y, (valor, marca, extremo) in enumerate(zip(datos[columna_valor], datos["nivel"], extremos)):
        punta = valor + extremo if valor >= 0 else valor - extremo
        ha = "left" if valor >= 0 else "right"
        despl = desplazamiento if valor >= 0 else -desplazamiento
        ax.text(punta + despl, y, f"{valor:.2f}{marca}", va="center", ha=ha, fontsize=8, color="#0b0b0b")

    ax.set_xlabel(etiqueta_x, color="#52514e", fontsize=8)
    ax.set_title(textwrap.fill(titulo, 55), color="#0b0b0b", fontsize=12)
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)


def preparar_datos_regresion(
    variables: pd.DataFrame, objetivo: str, umbral_nulos: float, umbral_identidad: float
):
    """Filtra columnas usables para explicar `objetivo` por regresion.

    Descarta: (a) columnas con demasiados nulos (se perderian muchas filas al
    hacer listwise deletion), y (b) columnas casi identicas al objetivo por
    construccion contable (ej. VAB estatal vs PIB estatal), que darian una
    significancia tautologica en vez de una relacion real.
    """
    candidatas = variables.drop(columns=[objetivo])

    nulos_frac = candidatas.isna().mean()
    descartadas_nulos = nulos_frac[nulos_frac > umbral_nulos].index.tolist()
    candidatas = candidatas.drop(columns=descartadas_nulos)

    corr_objetivo = variables[candidatas.columns.tolist() + [objetivo]].corr()[objetivo]
    descartadas_identidad = corr_objetivo.drop(objetivo)
    descartadas_identidad = descartadas_identidad[descartadas_identidad.abs() >= umbral_identidad]
    candidatas = candidatas.drop(columns=descartadas_identidad.index.tolist())

    datos = pd.concat([variables[objetivo], candidatas], axis=1).dropna()
    return datos[objetivo], datos[candidatas.columns], descartadas_nulos, descartadas_identidad


def podar_por_vif(x: pd.DataFrame, umbral_vif: float):
    x = x.copy()
    eliminadas = []
    while x.shape[1] > 1:
        x_const = sm_api.add_constant(x)
        vifs = pd.Series(
            [variance_inflation_factor(x_const.values, i) for i in range(1, x_const.shape[1])],
            index=x.columns,
        )
        peor_variable, peor_vif = vifs.idxmax(), vifs.max()
        if peor_vif <= umbral_vif:
            break
        eliminadas.append((peor_variable, round(float(peor_vif), 1)))
        x = x.drop(columns=[peor_variable])
    return x, eliminadas


def ajustar_regresion_estandarizada(y: pd.Series, x: pd.DataFrame):
    y_z = (y - y.mean()) / y.std()
    x_z = (x - x.mean()) / x.std()
    x_z = sm_api.add_constant(x_z)
    return sm_api.OLS(y_z, x_z).fit()


def tabla_regresion(modelo) -> pd.DataFrame:
    tabla = pd.DataFrame(
        {
            "coef_estandarizado": modelo.params,
            "error_estandar": modelo.bse,
            "t": modelo.tvalues,
            "p_valor": modelo.pvalues,
        }
    ).drop(index="const")
    tabla["significativo_5pct"] = tabla["p_valor"] < NIVEL_SIGNIFICANCIA
    tabla["nivel"] = tabla["p_valor"].apply(estrellas)
    tabla = tabla.reindex(tabla["t"].abs().sort_values(ascending=False).index)
    tabla.index.name = "variable"
    return tabla


def nombre_archivo(variable: str) -> str:
    limpio = variable.split("(")[0].strip()
    return "".join(c if c.isalnum() else "_" for c in limpio).strip("_").lower()


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
    parser.add_argument(
        "--variables-foco",
        default="IED,PIB (M MXN precios del 2018)",
        help="Variables (separadas por coma) para el analisis puntual de significancia",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Barras a mostrar por variable de foco")
    parser.add_argument(
        "--objetivo-regresion",
        default="PIB (M MXN precios del 2018)",
        help="Variable a explicar via regresion multiple. Vacio ('') para omitir este analisis",
    )
    parser.add_argument(
        "--umbral-identidad",
        type=float,
        default=0.98,
        help="Correlacion con el objetivo por encima de la cual una variable se considera casi-identica y se excluye",
    )
    parser.add_argument(
        "--umbral-vif",
        type=float,
        default=10.0,
        help="VIF maximo tolerado; se poda iterativamente la variable con mayor VIF hasta cumplir",
    )
    parser.add_argument(
        "--umbral-nulos",
        type=float,
        default=0.05,
        help="Fraccion maxima de nulos permitida en una variable candidata a la regresion",
    )
    parser.add_argument("--salida-dir", default=Path("salida_correlacion"), type=Path)
    args = parser.parse_args()

    args.salida_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(args.archivo, args.hoja)
    df = limpiar_columnas(df)
    variables = construir_matriz_variables(df, args.incluir_anio)

    corr, pval, nobs = calcular_correlacion_y_significancia(variables, args.metodo)
    corr.to_csv(args.salida_dir / "correlacion_ied.csv")
    pval.to_csv(args.salida_dir / "pvalores_ied.csv")

    graficar_heatmap_general(corr, pval, args.salida_dir / "correlacion_ied.png")

    total_pares = (pval.shape[0] * (pval.shape[0] - 1)) // 2
    significativos = int(((pval.values < NIVEL_SIGNIFICANCIA) & ~np.eye(len(pval), dtype=bool)).sum() / 2)
    print(
        f"Panorama general: {significativos} de {total_pares} pares de variables son "
        f"significativos al {int(NIVEL_SIGNIFICANCIA*100)}% ({significativos/total_pares:.1%}).\n"
    )

    for nombre_crudo in [v.strip() for v in args.variables_foco.split(",") if v.strip()]:
        variable = resolver_columna(nombre_crudo, corr.columns)
        tabla = tabla_detalle_variable(variable, corr, pval, nobs)
        archivo_base = nombre_archivo(variable)
        tabla.round(4).to_csv(args.salida_dir / f"detalle_{archivo_base}.csv")
        graficar_barras_significancia(
            tabla,
            columna_valor="correlacion",
            titulo=f"Variables mas correlacionadas con {variable}",
            etiqueta_x=(
                f"Correlacion ({int(NIVEL_SIGNIFICANCIA*100)}% signif.: * p<.05  ** p<.01  *** p<.001; "
                "barras atenuadas = no significativas)"
            ),
            salida=args.salida_dir / f"detalle_{archivo_base}.png",
            top_n=args.top_n,
        )

        n_sig = int(tabla["significativo_5pct"].sum())
        print(f"=== {variable} ===")
        print(f"{n_sig} de {len(tabla)} variables correlacionan de forma significativa (p<0.05).")
        print("Top 10 por fuerza de relacion:\n")
        print(tabla[["correlacion", "p_valor", "nivel"]].head(10).round(3).to_string())
        print()

    if args.objetivo_regresion:
        objetivo = resolver_columna(args.objetivo_regresion, variables.columns)
        y, x, descartadas_nulos, descartadas_identidad = preparar_datos_regresion(
            variables, objetivo, args.umbral_nulos, args.umbral_identidad
        )
        x_podada, eliminadas_vif = podar_por_vif(x, args.umbral_vif)
        modelo = ajustar_regresion_estandarizada(y, x_podada)
        tabla_modelo = tabla_regresion(modelo)

        archivo_base = nombre_archivo(objetivo)
        tabla_modelo.round(4).to_csv(args.salida_dir / f"regresion_{archivo_base}.csv")
        graficar_barras_significancia(
            tabla_modelo,
            columna_valor="coef_estandarizado",
            titulo=f"Que explica {objetivo} - regresion multiple estandarizada",
            etiqueta_x=(
                "Coeficiente estandarizado, IC 95% (* p<.05 ** p<.01 *** p<.001; "
                "barras atenuadas = no significativas)"
            ),
            salida=args.salida_dir / f"regresion_{archivo_base}.png",
            top_n=args.top_n,
            columna_error="error_estandar",
        )

        print(f"=== Regresion multiple: que explica {objetivo} ===")
        print(f"N observaciones: {int(modelo.nobs)}  |  Variables candidatas iniciales: {x.shape[1]}")
        print(
            f"Excluidas por nulos (> {args.umbral_nulos:.0%}): {len(descartadas_nulos)}  |  "
            f"Excluidas por casi-identicas al objetivo (r >= {args.umbral_identidad}): "
            f"{len(descartadas_identidad)}"
        )
        if len(descartadas_identidad):
            print("  " + ", ".join(descartadas_identidad.index.tolist()))
        print(f"Excluidas por colinealidad (VIF > {args.umbral_vif}): {len(eliminadas_vif)}")
        if eliminadas_vif:
            print("  " + ", ".join(f"{v} (VIF={vif})" for v, vif in eliminadas_vif))
        print(f"\nModelo final: {x_podada.shape[1]} variables independientes")
        print(f"R2 = {modelo.rsquared:.3f}  |  R2 ajustado = {modelo.rsquared_adj:.3f}  |  p-valor F = {modelo.f_pvalue:.2e}\n")
        print(tabla_modelo[["coef_estandarizado", "error_estandar", "t", "p_valor", "nivel"]].round(3).to_string())
        print()

    print(f"Resultados en: {args.salida_dir.resolve()}")


if __name__ == "__main__":
    main()
