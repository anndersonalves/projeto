# Detecção de fraude em cartão de crédito

Projeto de classificação em **dados reais de transações de cartão de crédito** (~284 mil transações, das quais só ~0,17% são fraude). O foco não é a acurácia, e sim **achar as fraudes** sem gerar um volume absurdo de falsos alarmes, usando recall, precisão, F1 e curva de precisão × recall da classe de fraude.

**Stack:** Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib.

## Como rodar

1. Abra `deteccao_fraude.ipynb` no [Google Colab](https://colab.research.google.com/) (Arquivo → Fazer upload de notebook, ou Arquivo → Abrir notebook → GitHub).
2. Execute tudo (Ambiente de execução → Executar tudo).

O dataset **não está no repositório** (o arquivo passa do limite de 100 MB do GitHub). O notebook lê direto do link:

```
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Para rodar localmente: `pip install -r requirements.txt` e abra o notebook no Jupyter (a primeira célula com `!pip install shap` pode ser ignorada).

## 1. O problema e por que o desbalanceamento muda a avaliação

Cada linha é uma transação. `Time` é o momento, `Amount` o valor, `Class` a resposta (1 = fraude, 0 = normal) e `V1` a `V28` são variáveis transformadas por PCA para proteger a privacidade dos clientes.

Dos 284.807 registros, só 492 são fraude (~0,172%). Um "modelo" que responde *"não é fraude"* para tudo acerta ~99,8% e **não encontra nenhuma fraude** (recall = 0). A acurácia, portanto, não mede o que importa aqui.

Por isso avalio sempre a **classe de fraude**:

- **Recall:** das fraudes reais, quantas o modelo achou (fraude que passa é prejuízo direto).
- **Precisão:** das transações marcadas como fraude, quantas eram mesmo (falso alarme é cliente com cartão bloqueado à toa).
- **F1:** equilíbrio entre as duas.
- **PR-AUC:** área da curva de precisão × recall, a mais honesta em dados desbalanceados, porque não é inflada pelos milhares de verdadeiros negativos como a curva ROC.

## 2. Preparação dos dados

- **Exploração:** sem valores nulos; proporção de fraudes medida com `value_counts(normalize=True)`.
- **`Amount`:** muito assimétrico, então criei `log_amount = np.log1p(Amount)` e descartei o original.
- **Split:** `train_test_split(test_size=0.2, stratify=y)`, para manter ~0,17% de fraudes em treino e teste (~98 fraudes no teste).
- **Padronização:** `StandardScaler` ajustado **só no treino**. Nos ajustes finais ele vai dentro de um `Pipeline`, para ser reajustado a cada dobra da validação cruzada, sem vazamento.
- **Desbalanceamento:** `class_weight="balanced"` (Random Forest, regressão logística) e `scale_pos_weight` (XGBoost). Under/oversampling foram testados à parte, **só no treino e depois do split**.

## 3. Comparação entre os modelos

Métricas da **classe de fraude** no conjunto de teste, limiar 0,5:

> **Preencher após rodar o notebook:** a última célula imprime essa tabela em Markdown pronta para colar aqui.

| Modelo | Precisão | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Regressão Logística (baseline) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| Regressão Logística (balanced) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| Random Forest (balanced) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| XGBoost (scale_pos_weight) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| XGBoost Pipeline + GridSearchCV | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

> **Preencher:** 2–3 frases sobre qual modelo foi melhor e por quê (compare o recall *e* a precisão, não só um deles), e o que a curva de precisão × recall mostrou que a ROC escondia.

## 4. Limiar de decisão e SHAP

O modelo devolve uma probabilidade de fraude e o limiar a transforma em decisão. Baixar de 0,5 para **0,3** marca mais transações como fraude: o recall sobe e a precisão cai.

| Limiar | Precisão | Recall | F1 | Fraudes pegas | Falsos alarmes |
|---|---|---|---|---|---|
| 0,5 | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| 0,3 | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

> **Preencher:** por que ficou com o limiar 0,3 (quantas fraudes a mais foram pegas, a que custo em falsos alarmes) e quais variáveis dominaram no SHAP (gráficos `beeswarm` e `bar`). Descreva também o exemplo de fraude pega e o de fraude perdida (gráficos `waterfall`). Lembre que `V1`–`V28` são anônimas: o SHAP mostra *quais* pesaram, não o significado delas.

## 5. O que mudei em relação ao que a Expert fez

> **Conferir antes de entregar:** a lista abaixo descreve o que há de diferente no meu notebook em relação ao roteiro do desafio. Ajuste conforme o que a Expert mostrou nas aulas.

- **Comparação de estratégias de balanceamento:** testei sem tratamento, pesos de classe, undersampling e oversampling, em regressão logística e XGBoost, em vez de escolher uma só (o enunciado sugere testar).
- **Regressão logística com e sem `class_weight`:** para isolar o efeito de dar peso à fraude.
- **GridSearchCV com várias métricas:** `refit="recall"`, mas guardando precisão, F1 e PR-AUC de cada combinação, porque escolher só pelo recall pode premiar um modelo que marca tudo como fraude.
- **Pipeline e grid também no XGBoost**, além da regressão logística.
- **Varredura de limiares (0,1 a 0,9)** e gráfico mostrando onde 0,5 e 0,3 caem na curva de precisão × recall, em vez de apenas trocar 0,5 por 0,3.
- **SHAP local:** waterfall de uma fraude pega e de uma fraude perdida, para explicar também os erros.
- **Cuidado com vazamento:** scaler ajustado só no treino e reamostragem só depois do split.

## Estrutura do repositório

```
deteccao-fraude-cartao/
├── deteccao_fraude.ipynb   # notebook completo, com saídas salvas
├── requirements.txt        # dependências
├── .gitignore              # impede subir o creditcard.csv
└── README.md
```

## Limitações

- Dados europeus de 2013: padrões de fraude mudam, então o modelo precisaria de retreino e monitoramento.
- A divisão é aleatória. Em produção, o correto seria dividir **por tempo** (treinar no passado, testar no futuro).
- Com ~98 fraudes no teste, as métricas oscilam bastante entre divisões e sementes.
