# Predicting Human Emotions Using AI from Brainwaves

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative" />
  <img src="https://img.shields.io/badge/Model-XGBoost-blue?logo=xgboost" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />  
  <img src="https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter" />
</p>

---

## Overview
This project develops an **AI-powered system** that predicts **human emotions** — such as **happiness**, **sadness**, and **fear** — from **EEG brainwave signals** using advanced **machine learning algorithms**.  
By analyzing neural activity patterns, the system classifies emotions in **real time**, helping **healthcare professionals** understand the emotional states of patients — especially those **unable to communicate** — thus enhancing **emotional awareness** and **patient-centered care**.

Multiple models were trained and compared:

- **Random Forest:** 72.5%  
- **AdaBoost:** 43.2%  
- **LightGBM:** 94.4%  
- **XGBoost (Final Model):** **96.3%**

The **XGBoost model** achieved the **highest accuracy of 96.3%**, with balanced performance across all emotion classes:  
**Precision = 0.96 | Recall = 0.96 | F1-Score = 0.96 | Accuracy = 96.3%**

---

## Dataset  

The dataset used in this project is sourced from a public EEG dataset containing recordings from **10 healthy participants**, designed specifically to analyze brain activity associated with **visually induced emotions**.

- **Emotions:** Happy 😊, Sad 😢, Fear 😨  
- **Channels:** 32-channel Emotiv Epoc Flex system (10-20 international standard)  
- **Sampling Rate:** 128 Hz  
- **Data Source:** [`EEG Emotion Recognition Dataset (GU)`](https://figshare.com/articles/dataset/EEG_Emotion_Recognition_Dataset_GU/29170289?file=54897092)  
- **Folder Used:** `Clean_Cliped/` (preprocessed EEG signals)  

The `Clean_Cliped` data was preprocessed by the original authors using a robust pipeline that included:  
- **Bandpass Filtering** – to isolate relevant EEG frequency bands  
- **Savitzky-Golay Smoothing** – to reduce high-frequency noise  
- **Independent Component Analysis (ICA)** – to remove eye-blink and motion artifacts  

---

## Methodology  

This project follows a structured machine learning pipeline to process and model EEG data for emotion recognition.

#### 1. Data Consolidation  
The raw dataset consisted of hundreds of individual `.mat` files (one per subject, emotion, and trial).  
A Python script (`eeg_final.ipynb`) was created to:
1. **Parse Filenames:** Extract `subject_id` and `emotion` labels directly from filenames.  
2. **Load Data:** Read EEG signals from each `.mat` file into NumPy arrays.  
3. **Combine Data:** Merge all records into a unified CSV file (`eeg_emotion_dataset.csv`) containing all samples with corresponding labels.

---

## Data Preparation for Machine Learning  
Before feeding the data to models, it underwent multiple preparation steps to ensure robustness:

- **Subject-Aware Splitting:**  
  Used `GroupShuffleSplit` to split data by `subject_id` to prevent data leakage — ensuring that EEG data from the same subject never appears in both training and testing sets.  

- **Windowing (Epoching):**  
  Continuous EEG signals were segmented into **2-second windows** (256 timesteps) with **50% overlap**, converting the time series into meaningful data samples.  

- **Data Division Strategy:**  
  - **95%** of the dataset was used for **model training and validation** (internally split into train/test).  
  - **5%** was held out as **real-world unseen data** for final model evaluation and deployment testing.  

---

## Model Architecture  

Rather than deep neural networks, this project employs a suite of powerful **tree-based ensemble machine learning models**.  
Each model was trained on extracted statistical and frequency-domain EEG features to classify the emotional state.

**Models Implemented:**
- **Random Forest:** Builds multiple decision trees and averages their predictions, reducing overfitting and improving robustness.  
- **AdaBoost:** Sequentially combines weak learners, giving more weight to previously misclassified samples.  
- **XGBoost (Extreme Gradient Boosting):** Highly optimized gradient boosting algorithm offering parallelization and regularization.  
- **LightGBM:** Efficient gradient boosting framework that grows trees leaf-wise, significantly speeding up training on large datasets.  

All models were trained and evaluated under the same preprocessing and feature extraction conditions to ensure fair comparison.  
The best-performing model was selected based on accuracy, precision, recall, and F1-score metrics.

---

*In summary, this dataset and preprocessing strategy allow reliable emotion classification from EEG brainwave signals, while maintaining subject independence and real-world generalization.*

## Features

- Real-time EEG signal processing and classification  
- Machine learning model based on XGBoost  
- Data preprocessing, cleaning, and visualization pipeline  
- Flask-based web interface for live emotion prediction  
- Modular, reproducible structure based on Cookiecutter Data Science  

---

## Project Structure

```bash
predicting-human-emotions-using-AI-from-brainwaves/
├── LICENSE
├── README.md
├── data
│   ├── raw/                # Original unprocessed EEG data
│   ├── interim/            # Intermediate transformed data
│   ├── processed/          # Final training and testing datasets
│       ├── train_dataset.csv
│       ├── real_world_test.csv
│       └── eeg_emotion_dataset.csv
│
├── models/                 # Trained ML models (e.g., XGBoost)
│   └── xgboost_model.pkl
│
├── notebooks/              # Jupyter Notebooks for exploration
│   └── eeg_final.ipynb
│
├── references/             # Dataset sources and documentation
│   └── data_source.txt
│
├── Deployment/             # Flask app and backend logic
│   └── app.py
│
├── reports/                # Generated reports and visualizations
│   └── figures/
│
├── requirements.txt        # Python dependencies
├── pyproject.toml
└── setup.cfg
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python ≥ 3.9  
- Pip ≥ 22.0  
- Jupyter Notebook  
- Dependencies listed in `requirements.txt`  

### Installation

```bash
# Clone repository
git clone https://github.com/Siddhanth2509/EEG-Brainwave-Analysis.git

# Navigate to project
cd predicting-human-emotions-using-AI-from-brainwaves

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the web app:

```bash
cd Deployment
streamlit run app.py
```

Or explore the model via Jupyter Notebook:

```bash
jupyter notebook notebooks/eeg_final.ipynb
```

---

## Model Details

- **Algorithm:** XGBoost Classifier  
- **Input:** EEG brainwave features (alpha, beta, theta, gamma bands)  
- **Output:** Predicted emotional state  
- **Metrics:** Accuracy, F1-score, Confusion Matrix  
- **Explainability:** SHAP visualizations for feature importance  

---

## 🗺️ Roadmap

- [x] Data preprocessing and feature extraction  
- [x] Model training and testing  
- [x] Flask deployment  
- [ ] Integration with real-time EEG hardware  
- [ ] Live dashboard for emotion tracking  

---

## Contributing

We welcome contributions!

1. Fork the repository  
2. Create a branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m "Added new feature"`  
4. Push to your fork: `git push origin feature-name`  
5. Open a Pull Request  

For detailed contribution steps, refer to `CONTRIBUTING.md`.

---

## License

Distributed under the [MIT License](LICENSE).  
Feel free to use and modify this project with attribution.

---

## Acknowledgments

- EEG Emotion Recognition dataset contributors  
- XGBoost and Scikit-learn open-source communities  
- Cookiecutter Data Science template  
- All contributors who supported this research  

---

<p align="center">
  <a href="#top"><img src="https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square" /></a>
</p>
