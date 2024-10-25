# Intrusion Detection System (IDS) Project
## Overview

This project focuses on developing an **Intrusion Detection System (IDS)** using machine learning techniques. The IDS aims to detect malicious network activities by classifying network traffic data as either benign or attack. The project involves data preparation, feature engineering, model training, evaluation, and comparison of different machine learning algorithms.

## Project Structure
- **Notebook**: `intrusion-detection-system.ipynb` - Contains the step-by-step development of the IDS, including code and detailed explanations
- **Data**: The dataset used for training and testing the models (Please refer to the Data section below)
- **Requirements**: `requirements.txt` - Lists all the dependencies and libraries required to run the project

## Features
- Data preprocessing and handling of class imbalance
- Implementation of various Machine Learning algorithms:
    - Random Forest
    - Logistic Regression
    - Decision Tree
- Evaluation using metrics like Accuracy, F1 Score, ROC AUC
- Visualization of model performance with ROC curves
- Hyperparameter tuning and cross-validation
- Adjusting classification thresholds and class weights
- Resampling techniques using SMOTE *(Synthetic Minority Oversampling Technique)*

## Dependencies
- Python 3.6 or higher
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- XGBoost

You can install all dependencies using the `requirements.txt` file provided

## Installation
1. Clone the repository
```
git clone https://github.com/rupert-x/Intrusion-Detection-System.git
cd Intrusion-Detection-System
```

2. Create a virtual environment (recommended):
```
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ids_env python=3.8
conda activate ids_env
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Download the dataset:
- Due to size constraints, the dataset is not included in the repository.
- Please download the dataset from the [CICIDS2017 dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) and place it in the `dataset/` directory

## Usage
1. **Run the Jupyter Notebook:**
```
jupyter notebook intrusion_detection_system.ipynb
```
2. **Follow the step in the notebook:**
    - The notebook is organized into sections with code cells and explanations
    - Run each cell sequentially to erproduce the results
    - Modify parameters or code as needed for experimentation

## Data
- **Dataset:** The *CICIDS2017* dataset is used, which contains both benign and attack network traffic data.
- Data Preprocessing:
    - Encoding Labels for binary classification
    - Handling missing values and infinite values
    - Feature Scaling using `StandardScaler`
    - Addressing class imbalance with `SMOTE`
## Results
- Model Evalution:
    - Performance metrics are calculated for each model, including Accuracy, F1 Score, and ROC AUC
    - ROC curves are plotted for visual comparison of model performance
- Best Model:
    - Based on the evaluation metrics, the best-performing model can be selected for deployment
## Dependencies Details:
Below is a list of the main libraries used in this project:
- **NumPy**: Fundamental package for numerical computations
- **Pandas**: Data manipulation and analysis library
- **Scikit-learn**: Machine Learning library for Python, used for model training and evaluation
- **Imbalanced-**learn: Library for handling imbalanced datasets, particularly SMOTE
- **Matplotlib**: Plotting library for creating static, animated, and interactive visualizations
- **Seaborn**: Data visualization library based on `Matplotlib`, provides a high-level interface for drawing attractive statistical graphics
- **XGBoost**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable
- **Jupyter Notebook**: An interactive computing environment for writing and running code
## Installation of Specific Libraries:
In case you need to install the libraries individually:
```
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost notebook
```
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgments
- **Dataset Source**: Canadian Institute for Cybersecurity (CICIDS2017)
- **Inspiration**: The need for robust intrusion detection systems in the field of cybersecurity and to gain practical hands on experience in designing a Machine Learning tool.
## Contact
For any questions or feedback, please contact:
- **Name**: Rob Lucian
- **Email**: lucian.robl@gmail.com
- **GitHub**: [rupert-x](https://github.com/rupert-x)