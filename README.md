# Hurricane Ian POI Visit Forecasting (D14 Task)

## 📌 Overview
This project studies the problem of **forecasting daily visits to Points of Interest (POIs)** in Florida during **Hurricane Ian (2022)**.  
The primary task is to **predict the 14th day’s visit count** at each POI, given the previous 13 days of observed visits.  

We experiment with **classical time-series models, deep learning methods, and large language models (LLMs)**, and extend them with **fine-tuning and retrieval-augmented generation (RAG)** to incorporate hurricane-specific contextual knowledge.

---

## 🗂 Data
- **Source:** SafeGraph visit patterns dataset  
- **Geographic scope:** Four Florida cities (Tampa, Miami, Orlando, Cape Coral)  
- **POI metadata:** placekey, city, location name, top category, latitude, longitude  
- **Temporal windows:** Before, during, Hurricane Ian landfall  
- **Forecasting setup:**  
  - Input sequence:  
    \\[
    v^p_{k-n+1:k-1} = [v^p_{k-13}, \\dots, v^p_{k-1}]
    \\]  
  - Forecast target:  
    \\[
    v^p_k \\quad (n=14)
    \\]

---

## ⚙️ Methods
We compared multiple approaches:  

1. **Classical Models**  
   - ARIMA  
   - Prophet  

2. **Deep Learning Models**  
   - LSTM  
   - RNN
   - GRU

3. **Large Language Models (LLMs)**  
   - Zero/few-shot forecasting with LLaMA-3.1-8B and Mistral-7B  
   - Fine-tuning with **LoRA** adapters for D14 forecasting  
   - **RAG-enhanced prompts** using hurricane-specific evacuation and landfall context  

---

## 🧪 Key Findings
- **Classical & deep models** perform reasonably but fail to adapt well when hurricane disruptions change visit patterns.  
- **LLMs without fine-tuning** show limited accuracy.  
- **Fine-tuned LLMs (LoRA)** substantially improve D14 predictions.  
- **RAG-enhanced prompts** (e.g., “Evacuation order announced 2 days ago”) allow LLMs to incorporate real-world context and further boost accuracy.  

---

## 📊 Evaluation
Metrics used:  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)
- RMSLE


Outputs include per-POI error analysis, top-10 best/worst cases, and visualizations.  

---

## 📂 Repository Structure
```
.
├── data/                  # Processed data files (train/test splits, JSONL, parquet)
├── scripts/               # Forecasting scripts for ARIMA, LSTM, LLMs
├── figures/               # Visualizations (forecast plots, model diagrams)
├── results/               # Evaluation outputs (CSV/plots per city/POI)
├── notebooks/             # Jupyter notebooks for exploratory analysis
└── README.md              # Project documentation
```

---

## 🚀 How to Run
1. Prepare train/test JSONL datasets (per city or multi-city).  
2. Run classical/deep baselines:  
   ```bash
   python arima_tampa_14.py --input ts_daily_panel/tampa_daily_panel.parquet --target d14
   ```  
3. Run LLM inference (example with LLaMA-3.1-8B):  
   ```bash
   python run_llama.py --input prepared_data/tampa_test.jsonl --output results/tampa_predictions.csv
   ```  
4. Evaluate results:  
   ```bash
   python evaluate.py --pred results/tampa_predictions.csv --metrics all
   ```

---

## ✨ Contributions
- Built a full **data preparation pipeline** for hurricane-aware POI visit forecasting.  
- Implemented and compared **classical, deep learning, and LLM-based approaches**.  
- Designed **generalized prompt templates** and **LoRA fine-tuning** for forecasting tasks.  
- Incorporated **geospatial RAG knowledge** (evacuation orders, landfall dates) into LLM forecasts.  

---


```
