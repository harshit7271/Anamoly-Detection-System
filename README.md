# Anomaly-Detection-System on Time Series Data - LSTM Autoencoder

Real-time anomaly detection on IoT-like time series using **TensorFlow/Keras LSTM Autoencoder**.  
Trained and evaluated on the **Numenta Anomaly Benchmark (NAB)** dataset.

---
Clone repo -
```bash
git clone https://github.com/harshit7271/Anamoly-Detection-System.git
cd Anamoly-Detection-System
code .
pip install -r requirements.txt
```


---
## 🚀 Project Highlights
- Implements an **LSTM Autoencoder** for sequence reconstruction
- Learns normal temporal patterns and flags deviations as anomalies
- Benchmarked on NAB for reproducibility and comparison

---

## Results
- Reconstruction error threshold dynamically set from training distribution
- Errors below threshold → Normal
- Errors above threshold → Anomaly
- Visualizations highlight anomalous points in time series

---

## Future Work
- Extend to other IoT datasets
- Experiment with GRU/LSTM hybrids
- Deploy as a streaming service with FastAPI
