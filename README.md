# 🏎️ Formula 1 Race Prediction System V2

An advanced machine learning system for predicting F1 race podium finishes using 33 engineered features, achieving 91.29% accuracy with interactive Streamlit deployment.

![Python](https://img.shields.io/badge/python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![Accuracy](https://img.shields.io/badge/Accuracy-91.29%25-brightgreen)

## 🚀 Live Demo

[**Try the App Here**](https://f1-predictor-v2-qo3y9pjrhoqxze7sswejcd.streamlit.app/)

## 📊 Project Overview

This project evolved from a basic 5-feature model (V1: 91.56% accuracy) to an advanced 33-feature system (V2: 91.29% accuracy with 0.947 ROC AUC) that incorporates:

- **Driver Performance Patterns**: Recent form, season statistics, reliability metrics
- **Circuit-Specific Intelligence**: Historical track performance, wins at circuit
- **Constructor Strength**: Team performance trends and recent results
- **Qualifying Analysis**: Grid position advantages and top qualification metrics
- **Track Characteristics**: Circuit length, downforce requirements, overtaking difficulty

## 🎯 Model Performance

| Metric | V1 (Baseline) | V2 (Advanced) | V2-Optimized |
|--------|---------------|---------------|--------------|
| **Accuracy** | 91.56% | 90.50% | **91.29%** |
| **ROC AUC** | N/A | 0.947 | **0.947** |
| **Features** | 5 | 33 | 33 |
| **Precision** | ~70% | 69.09% | **74.00%** |
| **Recall** | ~65% | 66.67% | **64.91%** |

## 🏗️ Architecture
```
f1-predictor-v2/
├── data/
│ ├── raw/ # Raw F1 race data (2020-2025)
│ ├── processed/ # Feature-engineered datasets
│ └── circuits/ # Track characteristic data
├── models/ # Trained XGBoost models (.pkl)
├── notebooks/ # Jupyter notebooks for development
├── mlruns/ # MLflow experiment tracking
├── streamlit_app.py # Interactive web application
└── requirements.txt # Python dependencies
```

## 🔧 Technologies

- **ML Framework**: XGBoost with hyperparameter optimization
- **Experiment Tracking**: MLflow
- **Data Source**: FastF1 API (Ergast F1 Database)
- **Web App**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 📈 Feature Engineering

### 33 Engineered Features Across 5 Categories:

1. **Driver Performance (11 features)**
   - Last 3/5 race average points
   - Season cumulative points and races
   - Podium and win rates
   - DNF rate and reliability metrics

2. **Circuit Intelligence (7 features)**
   - Historical wins at circuit
   - Average finish position at circuit
   - Circuit-specific win rate
   - Last year's finish at circuit

3. **Constructor Performance (5 features)**
   - Team season points
   - Last 5 races average points
   - Season race count

4. **Qualifying (5 features)**
   - Grid position
   - Qualified top 5/10 flags
   - Grid advantage metric

5. **Track Characteristics (7 features)**
   - Circuit length and corner count
   - Downforce level
   - Overtaking difficulty
   - Tire wear intensity

## 🚀 Quick Start

### Installation

Clone repository
git clone https://github.com/YOUR_USERNAME/f1-predictor-v2.git
cd f1-predictor-v2

Install dependencies
pip install -r requirements.txt

Run Streamlit app
streamlit run streamlit_app.py

### Using the App

1. **Select Circuit**: Choose from upcoming races (Mexico, Brazil, Vegas, Qatar, Abu Dhabi)
2. **Grid Positions**: Use championship order or enter custom qualifying results
3. **Get Predictions**: Click "Predict Top 3" for podium probability predictions

## 📊 Model Training Pipeline

The complete development process is documented in Jupyter notebooks:

1. **Data Collection**: FastF1 API integration (2020-2025 seasons)
2. **Feature Engineering**: 33 features from driver/circuit/team data
3. **Model Training**: XGBoost with 200 estimators, depth=7
4. **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold CV
5. **MLflow Tracking**: All experiments logged with parameters and metrics

## 🎯 Key Features

- ✅ **Interactive Web Interface**: Simple circuit selection and grid position entry
- ✅ **Real-time Predictions**: Podium probability for all 20 drivers
- ✅ **Circuit Intelligence**: Leverages historical track-specific performance
- ✅ **MLflow Integration**: Complete experiment tracking and model versioning
- ✅ **Production Ready**: Saved models, clean code structure, documentation

## 📝 Example Predictions

**Mexico City GP (Round 20) - Championship Order**:
- 🥇 **1st Place**: Oscar Piastri (McLaren) - 85.3% podium probability
- 🥈 **2nd Place**: Lando Norris (McLaren) - 47.5% podium probability  
- 🥉 **3rd Place**: George Russell (Mercedes) - 30.3% podium probability

## 🔮 V3 Roadmap

Planned enhancements for the next version:

- 🌦️ **Weather Integration**: Rain probability, temperature, track conditions
- 🛞 **Tire Strategy**: Compound performance, pit stop optimization
- 📻 **Practice Session Data**: FP1/FP2/FP3 pace analysis
- 🏁 **Safety Car Probability**: Historical SC rates per circuit
- 📊 **Team Radio Sentiment**: NLP analysis of team communications

## 📚 Dataset Information

- **Training Data**: 2,138 races (2020-2024 seasons)
- **Test Data**: 379 races (2025 season through Round 19)
- **Data Currency**: Through 2025 US GP (Round 19)
- **Next Update**: After Mexico GP (Round 20)


## 📄 License

MIT License - See LICENSE file for details


## 🙏 Acknowledgments

- FastF1 API for comprehensive F1 data access
- Ergast F1 Database for historical race statistics
- Formula 1 community for inspiration

---

**Built with ❤️ for F1 fans and ML enthusiasts**
