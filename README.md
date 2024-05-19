# LAB3
This report provides an overview of the work done in the two Jupyter notebooks focused on language modeling: one for regression tasks and the other for classification tasks.
## 1. Language Modeling - Regression
### Overview
The notebook demonstrates how to apply language modeling techniques to perform regression analysis on a given dataset.
### Objectives
The primary objectives were to implement preprocessing steps for text data, train various regression models, and evaluate and interpret the performance of these models.
### Methodology
The methodology began with data preprocessing, the dataset was loaded and initially explored. The text data was cleaned and preprocessed through tokenization, stopword removal, and stemming or lemmatization. This data was then transformed into numerical representations suitable for regression, such as TF-IDF and word embeddings.

For model training, the data was split into training and testing sets. Various regression models, including linear regression, and decision tree regression, were trained.

Evaluation of the models was carried out using metrics like Mean Squared Error (MSE) and R-squared, and the performance of different models was compared. Interpretation involved analyzing the results to understand the impact of different features on the regression outcome.
### Result
After evaluating the performances of Support Vector Regressor (SVR), Linear Regression, and Decision Tree models, several important conclusions can be drawn. Firstly, in terms of average prediction accuracy, SVR exhibits the lowest Mean Squared Error (MSE) among the three models, closely followed by the Decision Tree. This suggests that both SVR and Decision Tree models better fit the data compared to Linear Regression, which has a significantly higher MSE. When examining the dispersion of prediction errors, measured by Root Mean Squared Error (RMSE), SVR and Decision Tree show relatively similar values, indicating similar dispersion of errors around the mean value. In contrast, Linear Regression demonstrates a notably higher RMSE, implying greater variability in prediction errors. Consequently, the choice of model depends on the specific project objectives, performance requirements, and data characteristics. In this case, although SVR and Decision Tree models exhibit relatively similar performances, the final model selection should be based on comprehensive evaluation considering various aspects including accuracy, complexity, and interpretability.

## 2. Language Modeling - Classification
### Overview
This notebook focuses on applying language modeling techniques to classification tasks, with the aim of classifying text data into predefined categories.
### Objectives 
The objectives were to implement preprocessing steps for text data, train various classification models, and evaluate and interpret the performance of these models.
### Methodology
The methodology involved data preprocessing, where necessary libraries were imported, the dataset was loaded and explored, and the text data was cleaned and preprocessed through tokenization, stopword removal, and stemming or lemmatization. The text data was transformed into numerical representations suitable for classification, such as TF-IDF and word embeddings.

For model training, the data was split into training and testing sets, and multiple classification models including logistic regression, support vector machines, and random forests were trained. Hyperparameters were tuned to optimize model performance.

The models were evaluated using metrics such as accuracy, precision, recall, and F1-score, and the performance of different models was compared. Interpretation involved analyzing the results to understand the impact of different features on the classification outcome.
### Result
During the classification tasks, it was observed that the use of Word2Vec embeddings resulted in poor performance, with an accuracy of only 50%. This accuracy suggests that the model was essentially guessing at random, which indicates that Word2Vec embeddings did not capture the relevant features needed for effective classification in this particular dataset. This underperformance could be due to several reasons, such as insufficient training data for the embeddings, inappropriate preprocessing steps, or the need for more advanced feature extraction techniques.
