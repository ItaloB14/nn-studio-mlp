# app.py
# -*- coding: utf-8 -*-
"""
App interativo de Redes Neurais (MLP) para Excel/CSV
----------------------------------------------------
• Carrega dataset (.xlsx/.xls/.csv)
• Escolha do alvo e da tarefa (Auto, Classificação, Regressão)
• Hiperparâmetros: camadas ocultas, ativação, solver, alpha, LR, épocas, batch, seed
• Splits configuráveis (treino/validação/teste) com estratificação (quando possível)
• Pré-processamento: imputação + one-hot para categóricas, padronização opcional
• Treino, avaliação e gráficos (confusão, residual, curva de loss e val-score)
• Exporta: pipeline (preprocess+modelo), modelo, preprocessador, métricas e previsões
• Previsão em novos dados no mesmo app

Execute com:
    streamlit run app.py
"""

from __future__ import annotations
import io
import math
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from joblib import dump

st.set_page_config(page_title="NN Studio (MLP) — Excel/CSV", page_icon="🧠", layout="wide")

# =========================
# Helpers
# =========================

def load_df(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = uploaded.name.lower()
    data = uploaded.read()
    buf = io.BytesIO(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(buf)
    elif name.endswith(".csv"):
        # tenta utf-8, depois ;/latin-1
        try:
            buf.seek(0); return pd.read_csv(buf)
        except Exception:
            buf.seek(0); return pd.read_csv(buf, encoding="latin-1", sep=";")
    else:
        raise ValueError("Formato não suportado. Use .xlsx, .xls ou .csv")

def infer_task(y: pd.Series) -> str:
    s = y.dropna()
    if pd.api.types.is_float_dtype(s) and s.nunique() > max(20, len(s)//20):
        return "Regressão"
    if pd.api.types.is_integer_dtype(s) and s.nunique() <= max(20, len(s)//20):
        return "Classificação"
    if s.dtype == "object" or pd.api.types.is_categorical_dtype(s):
        return "Classificação"
    return "Regressão"

def parse_hidden_layers(text: str) -> Tuple[int, ...]:
    try:
        parts = [int(p.strip()) for p in text.split(",") if p.strip() != ""]
        if not parts:
            return (64, 32)
        parts = [max(1, min(2048, p)) for p in parts]
        return tuple(parts)
    except Exception:
        return (64, 32)

def make_preprocessor(df: pd.DataFrame, target_col: str, scale_numeric: bool):
    X = df.drop(columns=[target_col])
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols

def pack_download_bytes(obj, filename: str):
    """Serializa para memória (joblib/txt/csv)."""
    if filename.endswith(".joblib"):
        bio = io.BytesIO()
        dump(obj, bio)  # joblib para buffer
        return bio.getvalue(), filename
    elif filename.endswith(".txt"):
        return (obj or "").encode("utf-8"), filename
    elif filename.endswith(".csv"):
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_csv(index=False).encode("utf-8"), filename
        raise ValueError("Para .csv, forneça um DataFrame")
    else:
        raise ValueError("Extensão não suportada")

# =========================
# UI — Sidebar
# =========================

st.sidebar.title("⚙️ Configurações")
st.sidebar.caption("Ajuste hiperparâmetros e splits.")

with st.sidebar.expander("Splits (devem somar 100%)", expanded=True):
    col_s1, col_s2, col_s3 = st.columns(3)
    train_pct = col_s1.number_input("Treino (%)", 0, 100, value=70, step=1)
    val_pct   = col_s2.number_input("Validação (%)", 0, 100, value=10, step=1)
    test_pct  = col_s3.number_input("Teste (%)", 0, 100, value=20, step=1)
    if train_pct + val_pct + test_pct != 100:
        st.warning("As porcentagens **devem somar 100**.", icon="⚠️")

with st.sidebar.expander("Arquitetura e Otimização", expanded=True):
    hidden_layers_text = st.text_input("Camadas ocultas", value="64,32", help="Ex.: 128,64,32")
    activation = st.selectbox("Ativação", ["relu", "tanh", "logistic", "identity"], index=0)
    solver     = st.selectbox("Solver", ["adam", "lbfgs", "sgd"], index=0)
    alpha      = st.number_input("Alpha (L2)", min_value=1e-8, max_value=1.0, value=1e-4, step=1e-4, format="%.6f")
    lr         = st.number_input("Learning rate inicial", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-4, format="%.6f")
    max_iter   = st.slider("Épocas (max_iter)", 50, 2000, 300, step=50)
    batch_size = st.number_input("Batch size", 1, 8192, value=64, step=1)
    early_stopping = st.checkbox("Early stopping", value=True)
    scale_numeric  = st.checkbox("Padronizar numéricos", value=True)
    seed       = st.number_input("Seed", 0, 1_000_000, value=42, step=1)

# =========================
# UI — Layout principal
# =========================

st.markdown("# 🧠 NN Studio — MLP para Excel/CSV")
st.caption("Carregue um dataset, configure a rede neural e rode o treinamento em etapas.")

tabs = st.tabs(["1️⃣ Dados", "2️⃣ Configurar", "3️⃣ Treinar", "4️⃣ Avaliar & Exportar", "5️⃣ Prever em novos dados"])

# ---- TAB 1: Dados ----
with tabs[0]:
    st.subheader("Upload e pré-visualização")
    uploaded = st.file_uploader("Envie um arquivo .xlsx, .xls ou .csv", type=["xlsx", "xls", "csv"])
    if uploaded:
        try:
            df = load_df(uploaded)
            st.session_state["df"] = df
            st.success(f"Arquivo carregado! Linhas: {len(df)} | Colunas: {len(df.columns)}")
            st.dataframe(df.head(500), use_container_width=True)
        except Exception as e:
            st.error(f"Falha ao ler o arquivo: {e}")

# ---- TAB 2: Configurar ----
with tabs[1]:
    st.subheader("Escolha do alvo e da tarefa")
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("👉 Primeiro, carregue um arquivo na aba **Dados**.")
    else:
        default_target = df.columns[-1]
        for cand in ["target","alvo","y","label","classe","class","saida","output","response","resposta"]:
            if cand in [c.lower() for c in df.columns]:
                default_target = [c for c in df.columns if c.lower() == cand][0]
                break

        target_col = st.selectbox("Coluna-alvo (y)", list(df.columns), index=list(df.columns).index(default_target))
        st.session_state["target_col"] = target_col

        task_choice = st.selectbox("Tarefa", ["Auto", "Classificação", "Regressão"], index=0)
        if task_choice == "Auto":
            task_auto = str(infer_task(df[target_col]))
            st.caption(f"Detecção automática sugerida: **{task_auto}**")

        with st.expander("Dicas rápidas"):
            st.write("""
            - Para **classificação**, garanta que o alvo seja discreto (texto, categorias ou inteiros com poucas classes).
            - Para **regressão**, use alvo numérico contínuo.
            - **Padronização** é recomendada para MLP.
            - **Early stopping** usa uma fração interna de validação.
            """)

# ---- TAB 3: Treinar ----
with tabs[2]:
    st.subheader("Treinamento do modelo")
    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")
    if df is None or df.empty or target_col is None:
        st.info("👉 Configure os **Dados** e o **Alvo** nas abas anteriores.")
    else:
        task_choice = st.session_state.get("task_choice", "Auto")  # fallback
        if st.button("🚀 Treinar modelo agora", type="primary", use_container_width=True,
                     disabled=(train_pct + val_pct + test_pct != 100)):
            with st.spinner("Treinando..."):
                task = task_choice if task_choice != "Auto" else infer_task(df[target_col])
                preproc, num_cols, cat_cols = make_preprocessor(df, target_col, scale_numeric)

                y_raw = df[target_col]; X_df = df.drop(columns=[target_col])
                stratify = y_raw if (task == "Classificação" and y_raw.nunique() > 1) else None

                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X_df, y_raw, test_size=test_pct/100.0, random_state=seed,
                    stratify=stratify if task == "Classificação" else None
                )

                if early_stopping:
                    X_train, y_train = X_trainval, y_trainval
                    validation_fraction = max(0.05, min(0.4, val_pct / max(1, (train_pct + val_pct))))
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_trainval, y_trainval,
                        test_size=val_pct / max(1, (train_pct + val_pct)),
                        random_state=seed,
                        stratify=y_trainval if task == "Classificação" else None
                    )
                    validation_fraction = None

                hidden = parse_hidden_layers(hidden_layers_text)

                if task == "Classificação":
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden,
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        batch_size=batch_size if solver in ("adam","sgd") else "auto",
                        learning_rate_init=lr,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction if early_stopping else 0.1,
                        n_iter_no_change=10,
                        random_state=seed,
                        shuffle=True,
                        verbose=False
                    )
                    model_kind = "classifier"
                else:
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden,
                        activation=activation if activation in ("identity","relu","tanh","logistic") else "relu",
                        solver=solver,
                        alpha=alpha,
                        batch_size=batch_size if solver in ("adam","sgd") else "auto",
                        learning_rate_init=lr,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction if early_stopping else 0.1,
                        n_iter_no_change=10,
                        random_state=seed,
                        shuffle=True,
                        verbose=False
                    )
                    model_kind = "regressor"

                pipe = Pipeline(steps=[("preprocess", preproc), ("model", model)])

                if early_stopping:
                    pipe.fit(X_train, y_train)
                else:
                    pipe.fit(X_train, y_train)
                    if model_kind == "classifier":
                        y_val_pred = pipe.predict(X_val)
                        st.write(f"Validação — **Acurácia**: {accuracy_score(y_val, y_val_pred):.4f} | "
                                 f"**F1-macro**: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
                    else:
                        y_val_pred = pipe.predict(X_val)
                        st.write(f"Validação — **R²**: {r2_score(y_val, y_val_pred):.6f} | "
                                 f"**RMSE**: {math.sqrt(mean_squared_error(y_val, y_val_pred)):.6f}")

                y_pred = pipe.predict(X_test)

                metrics_lines = []
                if model_kind == "classifier":
                    acc = accuracy_score(y_test, y_pred)
                    f1m = f1_score(y_test, y_pred, average="macro")
                    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    metrics_lines += [
                        f"Acurácia (teste): {acc:.4f}",
                        f"F1-macro (teste): {f1m:.4f}",
                        f"Precisão-macro (teste): {prec:.4f}",
                        f"Revocação-macro (teste): {rec:.4f}",
                    ]

                    try:
                        proba = pipe.predict_proba(X_test)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            auc = roc_auc_score(y_test, proba[:, 1])
                            metrics_lines.append(f"ROC-AUC (binário, teste): {auc:.4f}")
                        else:
                            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                            metrics_lines.append(f"ROC-AUC (multiclasse OVR, macro, teste): {auc:.4f}")
                    except Exception:
                        metrics_lines.append("ROC-AUC indisponível (modelo não fornece probabilidades).")

                    try:
                        import numpy as np, matplotlib.pyplot as plt
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
                        ax.imshow(cm)
                        ax.set_title("Matriz de Confusão (teste)")
                        ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
                        for (i, j), v in np.ndenumerate(cm):
                            ax.text(j, i, str(v), ha="center", va="center")
                        st.pyplot(fig_cm, use_container_width=True)
                    except Exception:
                        pass

                    try:
                        rep = classification_report(y_test, y_pred, zero_division=0)
                        metrics_lines.append("\nRelatório de Classificação:\n" + rep)
                    except Exception:
                        pass

                    df_pred = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
                    if "proba" in locals():
                        if proba.ndim == 2:
                            for i in range(proba.shape[1]):
                                df_pred[f"proba_{i}"] = proba[:, i]

                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = math.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    metrics_lines += [
                        f"MAE (teste): {mae:.6f}",
                        f"MSE (teste): {mse:.6f}",
                        f"RMSE (teste): {rmse:.6f}",
                        f"R² (teste): {r2:.6f}",
                    ]

                    fig_sc, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, y_pred)
                    ax.set_title("Dispersão: Real vs Predito (teste)")
                    ax.set_xlabel("Real"); ax.set_ylabel("Predito")
                    st.pyplot(fig_sc, use_container_width=True)

                    resid = y_test - y_pred
                    fig_hist, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(resid, bins=30)
                    ax.set_title("Distribuição dos Resíduos (teste)")
                    ax.set_xlabel("Resíduo (y_real - y_pred)"); ax.set_ylabel("Contagem")
                    st.pyplot(fig_hist, use_container_width=True)

                    df_pred = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

                try:
                    model_obj = pipe.named_steps["model"]
                    if hasattr(model_obj, "loss_curve_") and len(model_obj.loss_curve_) > 1:
                        fig_loss, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(range(1, len(model_obj.loss_curve_) + 1), model_obj.loss_curve_)
                        ax.set_title("Curva de Perda (loss) por Época")
                        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
                        st.pyplot(fig_loss, use_container_width=True)
                except Exception:
                    pass

                try:
                    if hasattr(model_obj, "validation_scores_") and len(model_obj.validation_scores_) > 1:
                        fig_val, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(range(1, len(model_obj.validation_scores_) + 1), model_obj.validation_scores_)
                        ax.set_title("Score de Validação por Época (early_stopping)")
                        ax.set_xlabel("Época"); ax.set_ylabel("Score de Validação")
                        st.pyplot(fig_val, use_container_width=True)
                except Exception:
                    pass

                st.session_state["pipeline"] = pipe
                st.session_state["model_kind"] = model_kind
                st.session_state["pred_df"] = df_pred

                metrics_text = "\n".join(metrics_lines)
                st.session_state["metrics_text"] = metrics_text

                st.success("Treinamento finalizado!")
                st.code(metrics_text or "Sem métricas.", language="text")

# ---- TAB 4: Exportar ----
with tabs[3]:
    st.subheader("Relatório & Exportação")
    pipe = st.session_state.get("pipeline")
    metrics_text = st.session_state.get("metrics_text")
    pred_df = st.session_state.get("pred_df")

    if pipe is None:
        st.info("👉 Treine um modelo na aba **Treinar** para liberar downloads.")
    else:
        st.markdown("**Relatório resumido:**")
        st.code(metrics_text, language="text")

        c1, c2, c3, c4 = st.columns(4)
        bytes_pipeline, fname_pipeline = pack_download_bytes(pipe, "mlp_pipeline.joblib")
        c1.download_button("⬇️ Baixar Pipeline (recomendado)", bytes_pipeline, file_name=fname_pipeline)

        bytes_model, fname_model = pack_download_bytes(pipe.named_steps["model"], "modelo_mlp.joblib")
        c2.download_button("⬇️ Baixar Modelo", bytes_model, file_name=fname_model)

        bytes_pre, fname_pre = pack_download_bytes(pipe.named_steps["preprocess"], "pipeline_preprocessamento.joblib")
        c3.download_button("⬇️ Baixar Pré-processamento", bytes_pre, file_name=fname_pre)

        bytes_metrics, fname_metrics = pack_download_bytes(metrics_text or "", "relatorio_metricas.txt")
        c4.download_button("⬇️ Baixar Relatório", bytes_metrics, file_name=fname_metrics)

        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            st.markdown("**Prévia das previsões (teste):**")
            st.dataframe(pred_df.head(200), use_container_width=True)
            bpred, fpred = pack_download_bytes(pred_df, "previsoes_teste.csv")
            st.download_button("⬇️ Baixar Previsões (CSV)", bpred, file_name=fpred)

# ---- TAB 5: Prever em novos dados ----
with tabs[4]:
    st.subheader("Previsões em novos dados")
    pipe = st.session_state.get("pipeline")
    target_col = st.session_state.get("target_col")
    if pipe is None:
        st.info("👉 Treine um modelo primeiro para habilitar previsões.")
    else:
        st.caption("Envie um novo arquivo **com as mesmas colunas de entrada** (sem a coluna-alvo).")
        new_file = st.file_uploader("Novo arquivo (.xlsx/.xls/.csv) para previsão", type=["xlsx","xls","csv"], key="newpred")
        if new_file:
            try:
                new_df = load_df(new_file)
                if target_col in new_df.columns:
                    st.warning(f"Removendo coluna alvo detectada: {target_col}")
                    new_df = new_df.drop(columns=[target_col])
                yhat = pipe.predict(new_df)
                out_df = new_df.copy()
                out_df["predicao"] = yhat
                st.dataframe(out_df.head(300), use_container_width=True)
                b, fn = pack_download_bytes(out_df, "previsoes_novos_dados.csv")
                st.download_button("⬇️ Baixar Previsões (novos dados)", b, file_name=fn)
            except Exception as e:
                st.error(f"Falha ao gerar previsões: {e}")
