# Air Quality Forecasting Using Hybrid GRU and Tabular Neural Network

## Project Overview

This project predicts air quality measurements using a hybrid deep learning architecture built with the Keras Functional API.

The model combines:

* GRU layers for learning temporal patterns from historical observations
* Tabular environmental and time-based features
* Cyclical encoding of temporal information using sine and cosine transformations

Unlike traditional approaches that rely solely on sequential data, this architecture leverages both historical trends and contextual information to improve forecasting performance.

---

## Results

The final model achieved the following performance on the test set:

| Metric                    | Value  |
| ------------------------- | ------ |
| R² Score                 | 75.78% |
| Mean Squared Error (MSE)  | 0.4395 |
| Mean Absolute Error (MAE) | 0.4865 |

These results demonstrate that the hybrid architecture successfully captures a significant portion of the variability in air quality measurements while maintaining relatively low prediction error.

---

## Dataset

The dataset contains historical air quality measurements collected over time along with temporal and environmental attributes.

---

## Data Preprocessing

The preprocessing pipeline includes:

* Missing value interpolation
* Datetime feature extraction
* Cyclical encoding of hourly information
* Feature scaling
* Sequence generation using sliding windows
* Chronological train-test split

---

## Model Architecture

### Sequential Branch

GRU-based network designed to capture temporal dependencies:

* GRU(64)
* GRU(32)

### Tabular Branch

Dense layers used to process engineered and contextual features.

### Fusion Layer

Outputs from both branches are concatenated and passed through fully connected layers to generate the final prediction.

This design allows the model to learn both:

* Temporal dynamics
* Static contextual relationships

simultaneously.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

---

## Key Concepts Demonstrated

* Time Series Forecasting
* Deep Learning Regression
* GRU Networks
* Keras Functional API
* Multi-Input Neural Networks
* Feature Engineering
* Cyclical Encoding
* Data Preprocessing Pipelines

---

## Future Improvements

* Hyperparameter optimization
* Comparison against XGBoost and Random Forest baselines
* Deployment using FastAPI
* Real-time air quality forecasting
