# MBTI Personality Prediction using Machine Learning
## Introduction
The Myers Briggs Type Indicator (MBTI) is a personality type system that divides everyone into 16 distinct personalities based on four dimensions, namely: Introversion (I) - Extroversion (E), Intuition (N) - Sensing (S), Thinking (T) - Feeling (F), Judging (J) - Perceiving (P). In this project, I have developed a MBTI personality classifier that uses machine learning models to predict a person’s personality based on the social media posts per user as input.

## Methodology
### Exploring the Dataset
The [dataset used](https://www.kaggle.com/datasets/datasnaek/mbti-type?select=mbti_1.csv) has:
- 8675 rows
- 2 columns
  - type
  - posts.

The data in column ‘post’ contains 50 recent social media posts for each user. There are 16 unique labels in column ‘type’ with no null
values, each representing 16 MBTI type indicators. The post column had paragraphs which required some natural language processing in order to
perform the task of model training.
<p  align="center">
  <img src="https://github.com/Aayush-Gangwar/Personality-Prediction/assets/101112022/c3d75f16-910b-4272-8c74-f891977a719f" alt="About MBTI"><br>
<i>(Distribution of personality type in dataset)</i>
</p>

### Preprocessing
This is performed in order to reduced the inconsistency in the data by removing terms which do not contribute much to the person's personality.
1) Converting data in post column to lowercase so that 2 identical words written in different letter cases can be interpreted as similar.
2) Removing URLs and links
3) Removing special characters like ' , ', ' | ', ' - ' etc. and numbers 
4) Removing extra spaces
5) Removing stopwords such as ‘for’, ‘them’, ‘you’ etc. using the `nltk` library.
6) Perform word Lemmatization i.e. grouping of words with the same purpose together (e.g. gone, going, went to go).
### Models implemented
There are various classification algorithms present out of which I have implemented the following:
- Multinomial Naive Bayes
- Random Forest Classifier
- LightGBM Classifier
- Logistic Regression
## Result & Analysis
Multinomial Naïve Bayes performed worst because of it’s poor assumption.
Further , we can see ensemble model like LGBM perform the best.
Accuracies of models can be increased by required hyperparameter tuning.
