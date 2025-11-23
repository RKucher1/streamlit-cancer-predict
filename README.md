# Breast Cancer Predictor

A machine learning-powered web application built with Streamlit that predicts whether a breast tumor is malignant or benign based on cell nuclei measurements.

## Overview

This application assists medical professionals in diagnosing breast cancer from tissue sample measurements. It uses a Logistic Regression model trained on breast cancer data to predict tumor classification and provides probability estimates for both benign and malignant outcomes.

**Note:** This app is designed to assist doctors in their diagnosis process and should not be used as a replacement for professional medical opinion.

## Features

- **Interactive Measurements**: Adjust 30 different cell nuclei measurements using sidebar sliders
- **Real-time Predictions**: Get instant predictions as you modify measurement values
- **Visual Analysis**: Radar chart visualization comparing mean, standard error, and worst values across 10 key features
- **Probability Scores**: View the probability of benign vs malignant classification
- **Clean UI**: Modern, responsive interface with custom styling

## Measurements

The app analyzes 30 features across three categories:
- **Mean values**: Average measurements of cell nuclei
- **Standard error (SE)**: Variability in measurements
- **Worst values**: Largest/worst measurements observed

Features analyzed:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RKucher1/streamlit-cancer-predict
cd streamlit-cancer-predict
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

To start the Streamlit web application:

```bash
streamlit run app/main.py
```

The app will open in your default web browser at `http://localhost:8501`

### Training the Model

If you need to retrain the model with updated data:

```bash
python Model/main.py
```

This will:
- Load and clean the data from `Data/data.csv`
- Train a Logistic Regression model
- Save the trained model and scaler to `Model/model.pkl` and `Model/scalar.pkl`
- Display accuracy metrics and classification report

## Project Structure

```
streamlit-cancer-predict/
├── app/
│   └── main.py              # Main Streamlit application
├── Model/
│   ├── main.py              # Model training script
│   ├── model.pkl            # Trained logistic regression model
│   └── scalar.pkl           # StandardScaler for input normalization
├── Data/
│   └── data.csv             # Breast cancer dataset
├── assets/
│   └── style.css            # Custom CSS styling
└── requirements.txt         # Python dependencies
```

## Dependencies

- numpy==1.24.3
- pandas==2.0.1
- pickle5==0.0.11
- plotly==5.15.0
- scikit_learn==1.3.0
- streamlit==1.23.1

## How It Works

1. **Data Processing**: The application loads breast cancer data and maps diagnoses (M=Malignant, B=Benign)
2. **User Input**: Measurements can be adjusted via sidebar sliders or connected from cytology lab equipment
3. **Normalization**: Input values are scaled using the pre-trained StandardScaler
4. **Prediction**: The Logistic Regression model predicts the tumor classification
5. **Visualization**: Results are displayed with probability scores and an interactive radar chart

## Model Performance

The Logistic Regression model is trained using:
- 80/20 train-test split
- StandardScaler for feature normalization
- Random state: 42 for reproducibility

Run `python Model/main.py` to see current accuracy and classification metrics.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is available for educational and medical assistance purposes.

## Disclaimer

This application is intended as a diagnostic aid tool only. Always consult with qualified medical professionals for proper diagnosis and treatment decisions.
