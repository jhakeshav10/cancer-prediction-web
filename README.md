# Streamlit Breat Cancer Prediction Web App

## Introduction

This project is a web application that predicts the likelihood of breast cancer based on user-provided medical data. It leverages machine learning models to provide predictions in real time. The app is built using **Streamlit**, a Python framework for building web applications, and uses a trained model to make predictions.

## Screenshots

![alt text](https://github.com/jhakeshav10/cancer-prediction-web/blob/master/ui.png)

## Demo

You can see the live demo of the application here: [Cancer Prediction Web App](https://breast-cancer-prediction-regression.streamlit.app/)

## Features
- Real-time predictions for cancer likelihood based on input data.
- Clean and simple UI for ease of use.
- Easy-to-deploy web application using Streamlit.

## Technologies Used
- **Programming Language**: Python
- **Framework**: Streamlit
- **Machine Learning Libraries**: Scikit-learn, Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model**: Logistic Regression, Random Forest

## Getting Started

### Prerequisites

To run this project, ensure that you have the following installed:
- Python 3.x
- Libraries listed in `requirements.txt` (You can install them using `pip`)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/jhakeshav10/cancer-prediction-web.git
   ```

2. Navigate to the project directory:

   ```bash
   cd streamlit-cancer-predict
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. After installing the dependencies, you can run the Streamlit application using the following command:

   ```bash
   streamlit run app.py
   ```

2. The app should now be running locally at `http://localhost:8501/`. Open this URL in your web browser to interact with the app.

### Project Structure

- `app.py`: This is the main file containing the Streamlit application code.
- `model/`: This folder contains the machine learning model used for predictions.
- `requirements.txt`: Contains the list of dependencies required to run the project.
- `data/`: Any dataset used for training the model can be stored here.
