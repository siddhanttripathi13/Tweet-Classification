# NLP Tweet Classification

This project focuses on classifying tweets using Natural Language Processing (NLP) techniques. The goal is to develop a model that can accurately categorize tweets as either relevant or irrelevant to a specific topic. The project utilizes various NLP concepts and techniques such as CountVectorizer, TfidfVectorizer, and word2vec for text embedding. Classification models are trained using the embedded text data, and their performance is evaluated using different evaluation metrics and a confusion matrix.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository or download the project files.

2. Install the required libraries by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Open the provided Python notebook to access the code and data.

## Dataset

The dataset used in this project consists of tweets with associated labels. Each tweet is labeled as either "Relevant" or "Irrelevant." The dataset contains the following columns:

- `text`: The text content of the tweet.
- `choose_one`: The label assigned to the tweet.
- `class_label`: Numeric representation of the label.

## Data Preprocessing

Before training a machine learning model, the data undergoes several preprocessing steps:

1. Importing necessary libraries for data processing and analysis.
2. Loading the dataset using Pandas.
3. Cleaning the text by removing noise and unwanted characters.
4. Tokenizing the text by splitting it into individual words.
5. Preprocessing steps like removing stopwords and creating cleaned tokens.

## Embedding Techniques

To convert the text data into a numerical representation suitable for machine learning models, two embedding techniques are used:

1. **CountVectorizer**: Converts the text into a sparse matrix, where each row represents a document and each column represents a word. The matrix contains the count of each word in each document.
2. **TfidfVectorizer**: Assigns a score to each word based on its frequency in a document and rarity across all documents.

## Model Training and Evaluation

The classification models are trained using the embedded text data and evaluated using various metrics. The steps involved are:

1. Splitting the data into training and test sets.
2. Fitting a Logistic Regression classifier on the training data.
3. Predicting the labels for the test data.
4. Evaluating the performance of the classifier using metrics like accuracy, precision, recall, and F1 score.
5. Visualizing the confusion matrix to analyze the model's predictions.

## Results

The trained classifier achieves a certain level of accuracy, precision, recall, and F1 score, indicating its performance in classifying tweets as relevant or irrelevant. The confusion matrix visualization helps understand the distribution of true positives, true negatives, false positives, and false negatives.

## Conclusion

This project demonstrates how NLP techniques can be applied to classify tweets based on their relevance. By utilizing text embedding and machine learning algorithms, it becomes possible to automate the categorization process, which can be useful in various applications such as sentiment analysis and information retrieval.
