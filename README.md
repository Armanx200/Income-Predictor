# 🌟 Income Predictor 🌟

Welcome to the **Income Predictor** repository! This project uses machine learning to predict income based on various features from the dataset. We have utilized Random Forest and Gradient Boosting algorithms to achieve this.

## 🚀 Overview

This project takes a dataset and preprocesses it to convert categorical data to numerical data. After splitting the data into training and testing sets, we train a Random Forest model and a Gradient Boosting model. The accuracy of the Random Forest model is **0.86**.

## 📊 Results

### Random Forest Accuracy: 0.86

![Feature Importances](https://github.com/Armanx200/Income-Predictor/blob/main/Figure.png)

## 🛠️ How to Use

1. Clone the repository:
    ```sh
    git clone https://github.com/Armanx200/Income-Predictor.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Income-Predictor
    ```
3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the predictor:
    ```sh
    python Income_Predictor.py
    ```

## 📁 File Structure

- `Income_Predictor.py`: Main script for training and evaluating the models.
- `adult.csv`: The dataset used for training.
- `requirements.txt`: List of required libraries for the project.
- `Figure.png`: Plot showing feature importances of the model.

## 💡 Features

- **Data Preprocessing**: Handles missing values and encodes categorical variables.
- **Model Training**: Trains both Random Forest and Gradient Boosting models.
- **Hyperparameter Tuning**: Uses GridSearchCV for finding the best hyperparameters.
- **Model Evaluation**: Provides accuracy, classification report, and confusion matrix.

## 🤖 Models Used

- Random Forest Classifier
- Gradient Boosting Classifier

## 📈 Performance Metrics

- **Random Forest**: 
    - Accuracy: 0.86
    - Detailed classification report and confusion matrix available in the output.
- **Gradient Boosting**:
    - Try running the script to check the performance metrics.

## 🔧 Future Enhancements

- Add more models to compare.
- Perform more extensive hyperparameter tuning.
- Implement advanced feature engineering techniques.

## 📬 Contact

For any questions or suggestions, feel free to reach out:

- GitHub: [Armanx200](https://github.com/Armanx200)

---

Made with ❤️ by Armanx200
