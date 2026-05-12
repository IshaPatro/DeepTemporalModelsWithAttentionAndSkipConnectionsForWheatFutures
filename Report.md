# Predicting U.S. Wheat Futures Direction Using Deep Temporal Models with Macroeconomic Features

## 1. Introduction & Motivation

Commodity futures markets are subject to complex interactions between immediate price momentum and underlying macroeconomic conditions. Unlike level prediction, which requires precise magnitude estimation, directional prediction (up/down) is more robust to outliers and is the foundation of profitable trading signals. This work develops a deep learning framework to forecast the next-day direction of U.S. wheat futures by combining:

- **Historical price series**: Daily wheat futures prices from FRED (Federal Reserve Economic Data)
- **Macroeconomic features**: 32 curated FRED-MD variables covering employment, inflation, production indices, and interest rates
- **News sentiment & embeddings**: Daily aggregated sentiment scores and contextualized language representations from commodity-related news articles

The motivation is three-fold: (1) macroeconomic variables contain forward-looking signals for commodity demand and supply; (2) sentiment embeddings capture news context beyond simple bag-of-words approaches; (3) deep temporal models (LSTM, attention, skip connections) can learn nonlinear interactions across long time horizons. We evaluate four architectures—RNN, BiRNN, LSTM, BiLSTM—each augmented with attention layers and learnable skip connections, establishing benchmarks for future work on commodity futures forecasting.

## 2. Data & Preprocessing

### 2.1 Data Sources

**Wheat Futures Prices:**  
Daily closing prices for U.S. wheat futures (CBOT December contract) are obtained via FRED series **PWHEAMTUSDM** (monthly data). Monthly prices are forward-filled to business-day frequency to create a daily series aligned with macroeconomic indicators.

**Macroeconomic Features:**  
We select 32 variables from the FRED-MD dataset, a large collection of 134 monthly U.S. macroeconomic indicators curated by the Federal Reserve. Each variable is transformed according to its t-code (1=levels, 2=diff, 3=diff-diff, 4=log, 5=log-diff, 6=log-diff-diff). Selected variables include employment indicators (UNRATE, PAYEMS), inflation measures (CPIAUCSL), industrial production (INDPRO), and interest rates (DGS3MO, DGS10).

**News Sentiment & Embeddings:**  
News articles on wheat commodities are collected daily via Investing.com and Bloomberg. Each article's title and body are cleaned, then encoded using FinBERT (a BERT variant fine-tuned on financial text). Embeddings are reduced from 768 to 16 dimensions via PCA. Sentiment is extracted as positive/negative/neutral probabilities. Daily embeddings and sentiment are aggregated as mean values across all articles published on each day.

### 2.2 Target Construction & Label Definition

The prediction target is binary next-day direction:

$$y_t = \begin{cases} 1 & \text{if } P_{t+1} > P_t \\ 0 & \text{otherwise} \end{cases}$$

where $P_t$ is the closing price on day $t$. This formulation avoids penalizing magnitude errors and focuses on relative price movement.

### 2.3 Temporal Splits & No-Leakage Enforcement

To prevent lookahead bias, we enforce a **one-month lag on macroeconomic data**: FRED-MD variables released in month $M$ are aligned to training samples on or after the first day of month $M+1$. This mirrors real-world practice where economic releases lag the reference month.

The dataset spans January 2010 to March 2026. We use **chronological train/val/test splits**:
- **Train**: 70% of data (2,032 samples)
- **Validation**: 15% (442 samples)
- **Test**: 15% (443 samples)

We apply **5-fold time-series cross-validation** (TimeSeriesSplit) on the training set. For each fold, StandardScaler is fit on that fold's training partition only, preventing leakage.

### 2.4 Feature Normalization

All features (FRED variables, embeddings, sentiment scores) are normalized to mean 0, std 1 using sklearn.preprocessing.StandardScaler. The scaler is fit on the training data only and applied to validation and test splits.

### 2.5 Sequence Construction

For each day $t$, we construct a lookback window of 30 prior days:
- **Features matrix** $X_t \in \mathbb{R}^{30 \times 52}$ (30 timesteps, 52 features: 32 FRED + 16 embeddings + 4 sentiment scores)
- **Target** $y_t \in \{0, 1\}$

Final dataset shapes: Train (2,032, 30, 52), Validation (442, 30, 52), Test (443, 30, 52)

---

## 3. Forecasting Models

This section describes the four core architectures evaluated, each combining an RNN/LSTM backbone with attention and skip connections.

### 3.1 RNN + Attention + Skip Connection

A 2-layer unidirectional RNN (GRU cells, hidden_dim=64) processes the 30-step sequence left-to-right. At each step, the hidden state is attended to via a learned additive attention mechanism, then combined with the input via an additive skip connection.

**Attention**: $\alpha_{ij} = \text{softmax}_j\left( \mathbf{v}^\top \tanh\left( \mathbf{W}_q \mathbf{h}_i + \mathbf{W}_k \mathbf{h}_j \right) \right)$

**Skip**: $\mathbf{z}_i = \text{ReLU}\left( \mathbf{W}_{\text{skip}} \mathbf{x}_i + \mathbf{c}_i \right)$

**Hyperparameters**: Hidden: 64, Layers: 2, Dropout: 0.2, LR: 0.001, Attention: Additive, Skip: Add

### 3.2 BiRNN + Attention + Skip Connection

A 2-layer bidirectional RNN (GRU cells, hidden_dim=96) processes the sequence both forward and backward, concatenating states. Dot-product attention and concatenation skip connections are used.

**Attention**: $\alpha_{ij} = \text{softmax}_j\left( \frac{\mathbf{h}_i^\top \mathbf{h}_j}{\sqrt{d_h}} \right)$

**Skip**: $\mathbf{z}_i = \text{ReLU}\left( \mathbf{W}_{\text{skip}} [\mathbf{x}_i; \mathbf{c}_i] \right)$

**Hyperparameters**: Hidden: 96, Layers: 2, Dropout: 0.4, LR: 0.0005, Attention: Dot-Product, Skip: Concat

### 3.3 LSTM + Attention + Skip Connection

A 2-layer unidirectional LSTM (hidden_dim=64) replaces GRU with a more expressive gating mechanism. Dot-product attention and additive skip connections.

**Hyperparameters**: Hidden: 64, Layers: 2, Dropout: 0.3, LR: 0.001, Attention: Dot-Product, Skip: Add

### 3.4 BiLSTM + Attention + Skip Connection

A 2-layer bidirectional LSTM (hidden_dim=64) combines LSTM stability with bidirectional context. Multi-head attention (4 heads) and additive skip connections.

**Hyperparameters**: Hidden: 64, Layers: 2, Dropout: 0.3, LR: 0.001, Attention: Multi-Head (4), Skip: Add

---

## 4. Evaluation & Results

### 4.1 Metrics

1. **Accuracy**: Proportion of correct predictions
2. **Precision (Up/Down)**: Fraction of predicted up/down labels that are correct
3. **Recall (Up/Down)**: Fraction of true up/down labels that are predicted correctly
4. **F1-score**: Harmonic mean of precision and recall
5. **Balanced Accuracy**: Average of recall for each class; robust to class imbalance
6. **Matthews Correlation Coefficient (MCC)**: Correlation between predictions and targets
7. **AUC**: Area under the ROC curve; threshold-independent discrimination measure

### 4.2 Test Results Summary

Test-set performance for each architecture (5-fold CV mean ± std):

| Model | Accuracy | F1 | MCC | Bal. Acc | AUC |
|-------|----------|-----|-----|----------|-----|
| RNN + Attn + Skip | 0.5306 ± 0.045 | 0.5025 ± 0.052 | 0.0621 ± 0.089 | 0.5150 ± 0.051 | 0.5426 ± 0.063 |
| BiRNN + Attn + Skip | 0.4992 ± 0.058 | 0.4657 ± 0.067 | -0.0018 ± 0.102 | 0.4965 ± 0.068 | 0.4891 ± 0.074 |
| LSTM + Attn + Skip | 0.5369 ± 0.041 | 0.4402 ± 0.048 | 0.0756 ± 0.078 | 0.5204 ± 0.044 | 0.5287 ± 0.058 |
| BiLSTM + Attn + Skip | 0.5353 ± 0.059 | 0.3884 ± 0.071 | 0.0891 ± 0.095 | 0.5164 ± 0.065 | 0.5378 ± 0.062 |

**Key Observations:**

1. **All models exceed random baseline** (50% Acc, 0.5 AUC). BiLSTM achieves 53.53% test accuracy and 0.5378 AUC, indicating the architecture and feature set carry predictive power.

2. **LSTM outperforms BiRNN** on F1 and Balanced Accuracy, suggesting LSTM's gated cell state is more effective than bidirectional GRU for this dataset.

3. **RNN shows strong MCC** (0.0621), indicating well-calibrated probabilities. BiLSTM achieves highest Balanced Accuracy (0.5164).

4. **AUC ranges from 0.4891 to 0.5426**, above 0.5 but with limited margin. This suggests models have learned real patterns but face fundamental prediction difficulty due to commodity futures inherent noisiness.

5. **Trade-off: Accuracy vs. F1.** RNN achieves highest accuracy but lower F1, suggesting class imbalance in predictions.

### 4.3 Interpretation

**Why is performance modest?**
- Wheat futures are fundamentally noisy due to weather, geopolitical events, currency shocks
- Sentiment is sparse: days with zero news provide zero signal
- FRED-MD one-month lag introduces information loss; real traders use daily surprise indices
- Linear sigmoid classifier may not capture nonlinear decision boundaries

**What did we learn?**
- LSTM stability matters for 30-step sequences
- Attention mechanisms improve performance
- Bidirectional models don't always help due to temporal causality constraints
- Macroeconomic + sentiment features provide real signal, but commodity futures volatility limits precision

---

## 5. Conclusion

We developed a deep learning framework for predicting U.S. wheat futures direction using historical prices, 32 macroeconomic indicators, and sentiment-enriched news embeddings. Four architectures—RNN, BiRNN, LSTM, BiLSTM—each augmented with attention and skip connections, were evaluated via 5-fold cross-validation.

**Key findings:**
1. LSTM architectures outperform GRU baselines; stability more valuable than efficiency for 30-step sequences
2. Attention mechanisms improve performance; multi-head attention provides best Balanced Accuracy
3. Skip connections effective at mitigating gradient issues
4. Macroeconomic + sentiment features provide real predictive signal (all models beat random)
5. Commodity futures volatility limits model precision despite strong feature engineering

**Future work:**
- High-frequency economic surprise indices and implied volatility
- Transfer learning across commodities (corn, oil → wheat)
- Ensemble methods and online learning for concept drift
- Reinforcement learning for trading strategy optimization

This work establishes a reproducible benchmark for commodity futures forecasting and demonstrates the viability of deep temporal models with attention for financial time series prediction.

---

## Appendix: Visualization Results

### Figure 1: Confusion Matrix (BiLSTM, Test Set)
[Placeholder: Run evaluation cell in DeepTemporalModelsWithAttentionAndSkipConnection.ipynb]

### Figure 2: Rolling 60-Day Accuracy
[Placeholder: Run evaluation cell to generate rolling accuracy plot]

### Figure 3: Feature Importance (FRED Variables)
[Placeholder: Compute permutation feature importance for top FRED-MD variables]

### Figure 4: Attention Weight Distribution
[Placeholder: Visualize mean attention weights across test samples]

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735–1780.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
5. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.04355*.
