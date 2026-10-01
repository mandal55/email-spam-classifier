# Email Spam Classifier

## Intern ID
CITS9341

## Project Overview
A Machine Learning project that classifies emails as Spam or Ham (not spam) using text classification. Built using Python, Scikit-learn (TF-IDF + Naive Bayes).

## Features
- Loads email text dataset (`data/emails.csv`)
- Converts email text into numerical features using TF-IDF
- Trains a Naive Bayes classifier
- Evaluates model using accuracy and classification report
- Predicts spam/ham for new sample emails

## Files
- `email_spam_classifier.py` — main script
- `data/emails.csv` — dataset of sample emails labeled spam/ham
- `requirements.txt` — Python dependencies

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python email_spam_classifier.py
   ```

## Tech Used
- Python
- Pandas
- Scikit-learn (TF-IDF, Naive Bayes)

## Author
Submitted as part of CodTech IT Solutions Internship Program.
