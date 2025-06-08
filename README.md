# Mental Health Support Predictor

The Mental Health Support Predictor is a machine learning-powered application built using Python and Streamlit. It helps assess whether an individual may need mental health support based on workplace and personal inputs.

---

## 🚀 Features

- 🧠 Predicts mental health treatment need using a trained model
- 📋 Interactive Streamlit form UI
- 🔄 Gender normalization & missing value handling
- 🧮 One-hot encoding for multi-class features
- 🌲 Random Forest model with GridSearchCV tuning
- 💾 Model + encoder stored with joblib
- 📈 Performance visualizations with seaborn and matplotlib

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

---

## 💻 How to Run

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
mental-health-predictor/
├── app.py                  # Streamlit UI code
├── mental_health_model.pkl # Trained ML model
├── encoder.pkl             # One-hot encoder
├── feature_names.pkl       # Model feature list
├── survey.csv              # Cleaned OSMI dataset
├── README.md               # Project documentation
```

---

## ⚙️ How It Works

1. User fills out the form on the UI.
2. Input is encoded using the saved encoder.
3. Data is matched to training feature structure.
4. Model returns a binary prediction:
   - 1 → Needs Support
   - 0 → No Immediate Need
5. The result is displayed on the screen.

---

## 🧪 Algorithms & Processing

- **Model**: Random Forest Classifier (`sklearn`)
- **Tuning**: GridSearchCV for best parameters
- **Encoders**:
  - LabelEncoder for binary fields
  - OneHotEncoder for multi-class fields
- **Preprocessing**:
  - Gender normalization (e.g., 'male', 'M' → Male)
  - Null value handling
  - Feature encoding and alignment

---

## 📊 Visualizations 

You can view:
- Confusion Matrix
- Top Feature Importances

These are included in the training notebook but not on the app interface.

---

## 📝 Notes

- This project is based on the OSMI Mental Health in Tech Survey.
- It is for educational/demo purposes — not medical or clinical use.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Feel free to fork, open issues, or submit pull requests. Any improvement is appreciated.
