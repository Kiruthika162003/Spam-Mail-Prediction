# Spam Mail Prediction Project

This project aims to build a machine learning model for spam mail prediction using the Enron Email Dataset. The goal is to classify emails as spam or non-spam based on their content.

## Dataset

The Enron Email Dataset is a widely used dataset for email-related tasks. It consists of a large collection of emails from the Enron Corporation, which collapsed in 2001. The dataset contains both spam and non-spam (ham) emails, making it suitable for training a spam mail prediction model.

## Method Used

The chosen method for this project is the Naive Bayes classifier. Naive Bayes is a probabilistic classifier that works well with text-based data. It assumes that the features (words) are conditionally independent, given the class label. Despite this simplifying assumption, Naive Bayes often performs well in text classification tasks.

## Steps Involved

1. **Data Loading**: The Enron Email Dataset is loaded into the project. This dataset typically includes email text and corresponding labels indicating spam or non-spam.

2. **Preprocessing**: The dataset is preprocessed to prepare it for model training. This step may involve cleaning the data, removing unnecessary characters, converting text to lowercase, and handling missing values if any.

3. **Feature Extraction**: Text data is converted into numerical features that machine learning algorithms can understand. In this project, we use the CountVectorizer from the Scikit-learn library to create a bag-of-words representation of the emails.

4. **Splitting Data**: The dataset is split into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance.

5. **Model Training**: The Naive Bayes classifier is trained on the training set. During training, the model learns the relationships between the extracted features and the corresponding spam or non-spam labels.

6. **Prediction**: The trained model is used to predict the labels of the emails in the testing set.

7. **Model Evaluation**: The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1 score. These metrics help assess how well the model is able to classify spam and non-spam emails.

8. **Deployment**: Once the model is trained and evaluated, it can be deployed in a production environment to classify incoming emails as spam or non-spam in real-time.

## Conclusion

Spam mail prediction is an important task in email filtering and security. By using the Enron Email Dataset and the Naive Bayes classifier, this project aims to build an effective model for accurately classifying spam and non-spam emails. The steps outlined above provide a general overview of the project workflow, from data loading and preprocessing to model training and evaluation.

