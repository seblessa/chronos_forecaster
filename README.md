# 📈 Chronos Forecaster 

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AutoGluon](https://img.shields.io/badge/AutoGluon-Timeseries-orange)

🚀 **Chronos** is **Amazon's Foundation Model** for time series forecasting. It enables easy and efficient forecasting for various time series applications.

---

## 🔧 Installation

You can install Chronos Forecaster using pip:

```bash
pip install chronos_forecaster
```

---

## 📌 Usage

Here’s how you can use `ChronosForecaster` in your project:

```python
from chronos_forecaster import ChronosForecaster
import pandas as pd

# Load your data
df = pd.read_csv("your_timeseries_data.csv")

# Initialize the forecaster
forecaster = ChronosForecaster(
    forecast_horizon=24,  # Number of time steps to predict
    datetime_col="date",  # Column containing timestamps
    target_col="value",  # Column to forecast
    item_id_col=None,  # Specify if multiple time series exist
    random_state=42, # Random seed for reproducibility
    finetune=True, # Fine-tune the model (takes much longer than inference)
)

# Generate predictions
predictions = forecaster.predict(df)
print(predictions)
```

### **Example Output**
| date | lower_bound | predicted_value | upper_bound |
|------|------------|----------------|------------|
| 2025-03-18 00:00:00 | 90.2 | 95.5 | 100.7 |
| 2025-03-18 01:00:00 | 91.0 | 96.3 | 102.1 |

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions, feel free to:
- Open an **Issue**
- Submit a **Pull Request**

---

## ⭐️ Support the Project

If you find this package useful, consider **starring** the repository on GitHub! 🌟
