# UrbanPollution-Net: A Deep Learning Approach to Air Quality Index (AQI) Forecasting

## 🌐 Project Overview
UrbanPollution-Net is a data science and deep learning project focused on performing comprehensive Exploratory Data Analysis (EDA) and building an optimization-driven Artificial Neural Network (ANN) to forecast the Air Quality Index (AQI) across multiple major Indian metropolises. By modeling daily atmospheric observations, this project highlights non-linear chemical interactions and establishes reliable pollutant tracking mechanisms to support environmental data science and public health initiatives.

## 📊 Dataset Profile & Structural Attributes
The underlying dataset monitors daily atmospheric metrics recorded across various highly populated Indian cities, tracking the following 12 key pollutant concentrations which serve as continuous features:

* **Particulate Matter:** `PM2.5`, `PM10` (Fine and coarse inhalable particles)
* **Gaseous Pollutants:** `NO`, `NO2`, `NOx` (Nitrogen oxides), `NH3` (Ammonia), `CO` (Carbon monoxide), `SO2` (Sulfur dioxide), `O3` (Ozone)
* **Volatile Organic Compounds (VOCs):** `Benzene`, `Toluene`, `Xylene`

---

## 🛠️ Data Pipeline & Architecture Workflow

### 1. Exploratory Data Analysis (EDA) & Diagnostics
* **Distribution Evaluation:** Examined skewness and mathematical variance profiles across different urban centers to isolate localized pollution trends.
* **Correlation Mapping:** Evaluated multi-collinearity boundaries among chemical compositions to observe how distinct precursor compounds collectively drive the final calculated AQI metrics.

### 2. Preprocessing & Feature Scalability
* **Feature Normalization:** To prevent higher-magnitude variables (like PM10) from dominating the learning process, all 12 pollutant features were processed using robust scaling transformations (`StandardScaler`). This guarantees that gradient updates propagate symmetrically through the deep learning layers.

### 3. Model Architecture (`UrbanPollution-Net`)
The core prediction framework relies on a fully connected, feedforward Sequential Artificial Neural Network (ANN) implemented via Keras and TensorFlow:

* **Input Layer:** Accepts the 12 scaled atmospheric feature nodes.
* **Hidden Layers:** Configured with dense layers utilizing the non-linear Rectified Linear Activation Function (`ReLU`) to capture complex, high-dimensional chemical dependencies.
* **Output Layer:** Comprises a single linear neuron optimized to predict the continuous, scalar target value of the `AQI`.

---

## 📈 Model Performance & Analytical Takeaways
* **Generalization Limits:** By enforcing a strict train-test validation split, the deep learning model successfully evaded overfitting limitations.
* **Prediction Accuracy:** The model achieved a robust **Root Mean Squared Error (RMSE) of approximately 21 points** on completely unseen evaluation data. This confirms that the network can reliably map raw chemical observations to standardized index benchmarks across varying urban environments.

## 💻 Tech Stack & Environment
* **Language Framework:** Python 3 (Jupyter Notebook)
* **Core Data Tools:** `pandas`, `numpy`
* **Visualization Layer:** `matplotlib`, `seaborn`
* **Deep Learning Engine:** `tensorflow.keras` (Sequential API)
