# 🧠 Employee Salary Prediction using Machine Learning

This project predicts employee salary categories (≤ ₹50,000 or > ₹50,000) based on demographic and work-related features using various machine learning algorithms. The goal is to help identify potential income categories from given attributes like age, education, occupation, etc.

---

## 📁 Project Structure

employee-salary-prediction/
│
├── employee salary prediction.ipynb # Main Jupyter Notebook with all ML code
├── salary.pkl # Trained machine learning model (if exported)
├── app.py # Optional Streamlit app (if created)
└── README.md # Project documentation

---

## 🚀 Features

- Data preprocessing (handling missing values, encoding, scaling)
- Model training with classifiers (e.g., CatBoost, MLPClassifier)
- Evaluation using accuracy and classification report
- Streamlit frontend integration (optional)
- Salary prediction displayed with estimated value and class

---

## 📊 Dataset

The dataset used is the [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult), commonly used for salary prediction tasks.

**Features include:**

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Race
- Sex
- Hours-per-week
- Capital gain/loss
- Native country

Target variable: `salary` (≤ ₹50,000 or > ₹50,000)

---

## 🛠️ Installation & Setup

1. Clone the repository:

git clone https://github.com/Anshul412/Employee_Salary_prediction.git
cd Employee_Salary_prediction

2.Install dependencies:

pip install -r requirements.txt
If no requirements.txt, install manually:
pip install pandas numpy scikit-learn catboost streamlit matplotlib seaborn

💡 Usage

➤ Jupyter Notebook
Open the notebook:

jupyter notebook "employee salary prediction.ipynb"
Run all cells to view the entire ML workflow.

➤ Streamlit App (Optional)
If you have an app.py:

streamlit run app.py
This will launch a simple web interface for salary prediction.

## Model Information

The application uses a Multi-Layer Perceptron (MLP) neural network classifier from scikit-learn. The model is trained on the Adult Census Income dataset with the following configuration:

- Solver: Adam optimizer
- Hidden Layer Sizes: (5, 2)
- Random State: 2
- Max Iterations: 2000

The model achieves approximately 84% accuracy on the test set.

## Data Preprocessing

The application performs the following preprocessing steps:

1. Removes certain rare categories (e.g., "Without-pay", "Never-worked")
2. Encodes categorical variables
3. Scales numerical features using MinMaxScaler

## Future Improvements

- Add more visualization options for the prediction results
- Implement feature importance analysis
- Add option to compare different models
- Improve the UI with more interactive elements

📈 Sample Output

Predicted Salary Class: ≤ ₹50,000
Estimated Salary: ₹26,953
📌 Future Improvements
Model optimization with GridSearchCV

Add SHAP/Feature importance visualizations

Export predictions to CSV

Dockerize the application

📚 License
This project is for educational purposes. If reused, please credit the original authors.

🙋‍♂️ Author
Anshul Gupta
