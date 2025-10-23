# 📈 Task 3: Linear Regression

## 📘 Objective
The goal of this task is to **implement and understand simple & multiple linear regression** models using Python.  
You’ll learn how to preprocess data, train a regression model, evaluate it using various metrics, and interpret results.

For this task, I used the **House Price Prediction Dataset** from Kaggle.  
📂 [Dataset Link](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

---

## 🧠 What I Did
1. Imported and preprocessed the dataset using **Pandas** and **NumPy**.  
2. Split the data into **training** and **testing** sets.  
3. Trained a **Linear Regression** model using `sklearn.linear_model`.  
4. Evaluated performance using **MAE**, **MSE**, and **R² score**.  
5. Visualized regression lines and interpreted model coefficients.  

---

## 🧰 Tools & Libraries Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

### 1. Data Preprocessing
- Loaded the dataset (`.csv` file) into a Pandas DataFrame.
- Checked for and handled missing values (if any).
- Encoded categorical features (e.g., converted 'yes'/'no' to 1/0, used One-Hot Encoding for `furnishingstatus`).
- Performed feature scaling (StandardScaler, MinMaxScaler) if necessary.

### 2. Train-Test Split
- Separated the data into features (X) and the target variable (y).
- Split the dataset into training and testing sets (e.g., 70% train, 30% test) to prevent overfitting.

### 3. Model Training
- Initialized the `LinearRegression` model from `sklearn.linear_model`.
- Trained (fit) the model on the training data (`X_train`, `y_train`).

### 4. Model Evaluation
- Used the trained model to make predictions on the unseen test data (`X_test`).
- Evaluated the model's performance by comparing predictions (`y_pred`) to actual values (`y_test`) using:
    - **MAE (Mean Absolute Error)**
    - **MSE (Mean Squared Error)**
    - **$R^2$ (R-squared) Score**

---

## 📊 Results & Visualization
[Showcase the outcome of your project. This is the "proof" of your work.]

### Key Metrics
Here is the performance of the model on the test set:
- **$R^2$ Score:** [Your $R^2$ Score, e.g., 0.6463]
- **Mean Absolute Error (MAE):** [Your MAE Score]
- **Mean Squared Error (MSE):** [Your MSE Score]

**Interpretation:** The $R^2$ score of [Your Score] indicates that approximately [Your Score * 100]% of the variance in the target variable can be explained by the features in the model.

### Visualization
[Include key plots. For regression, an "Actual vs. Predicted" plot is standard.]

This plot shows the actual prices from the test set (X-axis) against the model's predicted prices (Y-axis). The red line represents a perfect prediction.

![Actual vs. Predicted Prices]([FILENAME_OF_YOUR_PLOT.png])

---

## ⚙️ How to Run
[Provide clear instructions for someone to run your project on their local machine.]

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[YOUR_REPO_NAME].git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd [YOUR_REPO_NAME]
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt`, list the libraries manually):*
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
4.  **Run the script:**
    ```bash
    python [YOUR_SCRIPT_NAME.py]
    ```

---
