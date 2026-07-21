# Phishing-Email-Detection-Using-Machine-Learning

A machine learning project that detects whether an email is a phishing email or a safe email using Natural Language Processing (NLP) and Scikit-learn.

## Features

- Detects phishing and safe emails
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Displays model accuracy
- Generates classification report
- Shows confusion matrix
- Predicts custom email text

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset

The dataset contains labeled phishing and safe emails.

Columns:
- Email Text
- Email Type

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python phishing_detection.py
```
## Dataset

The dataset is not included in this repository because of its size.

Download the dataset and save it as `emails.csv` in the project folder before running the project.

## Sample Output

```
Model Accuracy: 92.94%

Prediction:
Phishing Email
```

## Project Structure

```
Phishing-Email-Detection/
│── phishing_detection.py
│── email.csv
│── requirements.txt
│── README.md
```

## Author

Umeir Patel**
