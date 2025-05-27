# 📈 Chronos Forecaster 

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![AutoGluon](https://img.shields.io/badge/AutoGluon-Timeseries-orange)

🚀 **Chronos** is **Amazon's Foundation Model** for time series forecasting, designed to handle a wide range of forecasting tasks with high accuracy and minimal effort. The model is trained on diverse real-world time series data and demonstrates strong generalization across domains.


🔗 **GitHub Repository**: [amazon-science/chronos](https://github.com/amazon-science/chronos-forecasting)  
 📄 **Research Paper**: [Chronos: Learning the Language of Time Series](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series)  

> 💡 **Why not just use Chronos from the official repo?**  
The official setup with AutoGluon requires multiple manual steps — like configuring the model, validating input formats, setting paths, and cleaning up after training.  `ChronosForecaster` handles all of that for you. It simplifies the process into a single, easy-to-use class so you can go from raw data to forecasts in just a few lines of code.


---


## 🔧 Installation

You can install Chronos Forecaster using pip:

```bash
pip install chronos_forecaster
```

---

## 📂 Input Data Format

Chronos expects a DataFrame with the following columns:

| date                | feature_1 | feature_2 | target |
|---------------------|-----------|-----------|--------|
| 2025-01-01 00:00:00 | ...       | ...       | 84.2   |
| 2025-01-01 01:00:00 | ...       | ...       | 86.1   |
| ...                 | ...       | ...       | ...    |

- **`date`**: Timestamp column (required)  
- **`target`**: Target column to forecast (required)  
- **Additional features** can be present in the dataset, but they are **ignored by Chronos**, as it is a **univariate forecasting model**.

If your dataset contains multiple time series, include an `item_id` column and pass its name to the `item_id_col` parameter.

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
    frequency = "h", # Frequency of the time series
    target_col="target",  # Column to forecast
    item_id_col=None,  # Specify if multiple time series exist
    random_state=42, # Random seed for reproducibility
    finetune=False, # Fine-tune the model (takes much longer than inference)
)

# Generate predictions
predictions = forecaster.predict(df)
print(predictions)
```

### **Example Output**
| date | lower_bound | target_predicted | upper_bound |
|------|------------|----------------|------------|
| 2025-03-18 00:00:00 | 90.2 | 95.5 | 100.7 |
| 2025-03-18 01:00:00 | 91.0 | 96.3 | 102.1 |
| ...                 | ...       | ...       | ...    |

> 🧠 **Note**: The `lower_bound` and `upper_bound` columns represent Chronos' **80% prediction interval**. This means there's an 80% chance the actual future value will fall within this range.

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
