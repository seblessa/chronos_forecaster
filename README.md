# 📈 Chronos Forecaster 

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AutoGluon](https://img.shields.io/badge/AutoGluon-Timeseries-orange)

🚀 **Chronos** is **Amazon's Foundation Model** for time series forecasting, designed to handle a wide range of forecasting tasks with high accuracy and minimal effort. The model is trained on diverse real-world time series data and demonstrates strong generalization across domains.

🔗 **GitHub Repository**: [amazon-science/chronos](https://github.com/amazon-science/chronos-forecasting)  
📄 **Research Paper**: [Chronos: Learning the Language of Time Series](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series)  

> 💡 **Why not just use Chronos from the official repo?**  
The official setup with AutoGluon requires multiple manual steps — like configuring the model, validating input formats, cleaning up logs, and cleaning up after inference. `ChronosForecaster` handles all of that for you. It simplifies the process into a single, easy-to-use class so you can go from raw data to forecasts in just a few lines of code.

**✨ New in v0.2.0:** Multi-engine support! Switch between **Chronos-Bolt** and **Chronos-2** with a single parameter. Both engines now use the unified `chronos-forecasting` library for zero-shot forecasting. Chronos-2 adds support for **past and future covariates** to improve forecasting accuracy.

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
- **Additional features** can be present in the dataset, but they are **ignored by Chronos**, as it is a **univariate forecasting model**. However, **Chronos-2 supports covariates** — see below for details.

If your dataset contains multiple time series, include an `item_id` column and pass its name to the `item_id_col` parameter.

---

## 📌 Usage

Here's how you can use `ChronosForecaster` in your project:

```python
from chronos_forecaster import ChronosForecaster
import pandas as pd

# Load your data
df = pd.read_csv("your_timeseries_data.csv")

# Initialize the forecaster
forecaster = ChronosForecaster(
    forecast_horizon=24,  # Number of time steps to predict
    datetime_col="date",  # Column containing timestamps
    frequency="h",  # Frequency of the time series
    target_col="target",  # Column to forecast
    item_id_col=None,  # Specify if multiple time series exist
    random_state=42,  # Random seed for reproducibility
    engine="chronos2",  # "chronos" for Chronos-Bolt, "chronos2" for Chronos-2
)

# Generate predictions
predictions = forecaster.predict(df)
print(predictions)
```

### **Example Output**
| date                | lower_bound | target_predicted | upper_bound |
|---------------------|-------------|------------------|-------------|
| 2025-03-18 00:00:00 | 90.2        | 95.5             | 100.7       |
| 2025-03-18 01:00:00 | 91.0        | 96.3             | 102.1       |
| ...                 | ...         | ...              | ...         |

> 🧠 **Note**: The lower_bound and upper_bound columns represent Chronos' 80% prediction interval. This means there's an 80% chance the actual future value will fall within this range.

---

## 🔧 Advanced Features

### **Using Covariates (Chronos-2 only)**

Chronos-2 can leverage additional features (covariates) to improve forecasting:

```python
# Past covariates: historical features aligned with training data
past_covariates = df[["date", "temperature", "holiday"]]

# Future covariates: known future values (e.g., weather forecasts, calendar features)
future_covariates = future_df[["date", "temperature", "holiday"]]

forecaster = ChronosForecaster(
    forecast_horizon=24,
    datetime_col="date",
    target_col="target",
    frequency="h",
    engine="chronos2",  # Must use chronos2 for covariates
)

predictions = forecaster.predict(
    df=df,
    past_covariates_df=past_covariates,
    future_covariates_df=future_covariates,
)
```

**⚠️ Important Notes:**
- Covariates are **only supported by the `chronos2` engine**
- Future covariates should only contain **truly known future values** to avoid data leakage
- When using multiple time series (`item_id_col`), covariates must include the item ID column
- Test a baseline without covariates first to verify they add value

### **Multiple Time Series (Panel Data)**

```python
forecaster = ChronosForecaster(
    forecast_horizon=24,
    datetime_col="date",
    target_col="sales",
    item_id_col="store_id",  # Column identifying different time series
    frequency="D",
    engine="chronos2",
)

predictions = forecaster.predict(panel_data)
# predictions will include 'store_id' to identify each series
```

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions, feel free to:
- Open an **Issue**
- Submit a **Pull Request**

---

## 🙏 Acknowledgments

This package builds upon the excellent work by Amazon Science:
- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821)

---

## ⭐️ Support the Project

If you find this package useful, consider **starring** the repository on GitHub! 🌟
