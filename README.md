# NN Studio — MLP para Excel/CSV (Streamlit)

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abra o link local que o Streamlit mostrar (geralmente http://localhost:8501).

## Funcionalidades
- Upload de .xlsx/.xls/.csv
- Detecção automática da tarefa (classificação/regressão) ou escolha manual
- Configuração de MLP (camadas, ativação, solver, alpha, LR, épocas, batch, seed)
- Splits treino/validação/teste com estratificação quando aplicável
- Pré-processamento (imputação, one-hot, padronização opcional)
- Métricas e gráficos (confusão, residual, loss, val-score)
- Exportação de pipeline, modelo, preprocessador, métricas e previsões
- Previsões em novos dados no próprio app
