# 📖 Complete NLP Learning

A comprehensive educational repository covering Natural Language Processing (NLP) fundamentals, techniques, and practical machine learning applications. This project progresses from basic tokenization concepts through advanced text analysis methods and real-world sentiment analysis projects.

## 📚 Overview

This repository provides hands-on Jupyter notebooks and machine learning projects demonstrating core NLP concepts and their practical applications. It's designed as a learning resource for anyone interested in understanding how text is processed and analyzed programmatically.

## 📁 Repository Structure

### Core Concepts - Fundamental NLP Tutorials

These notebooks cover essential NLP preprocessing and analysis techniques:

1. **[01_Tokenization_using_NLTK.ipynb](01_Tokenization_using_NLTK.ipynb)** - Introduction to breaking text into tokens (words, sentences, characters)
2. **[02_Stemming.ipynb](02_Stemming.ipynb)** - Reducing words to their root form using stemming algorithms
3. **[03_Lemmatization.ipynb](03_Lemmatization.ipynb)** - Morphological word normalization with lemmatization
4. **[04_Stopwords.ipynb](04_Stopwords.ipynb)** - Identifying and removing common non-meaningful words
5. **[05_Parts_of_speech_Tagging.ipynb](05_Parts_of_speech_Tagging.ipynb)** - Tagging words with their grammatical roles (noun, verb, etc.)
6. **[06_Named_entity_recognition.ipynb](06_Named_entity_recognition.ipynb)** - Extracting named entities (persons, organizations, locations)
7. **[07_Bag_of_words.ipynb](07_Bag_of_words.ipynb)** - Text vectorization using the Bag of Words model
8. **[08_TF-IDF.ipynb](08_TF-IDF.ipynb)** - Term Frequency-Inverse Document Frequency for feature extraction
9. **[09_Word2Vec_implementation.ipynb](09_Word2Vec_implementation.ipynb)** - Dense word embeddings and semantic relationships

### Machine Learning Projects

Applied projects that use the concepts above for real-world NLP tasks:

1. **[Projects/01_SpamHam_using_BOW_ML.ipynb](Projects/01_SpamHam_using_BOW_ML.ipynb)** - SMS spam classification using Bag of Words
2. **[Projects/02_SpamHam_using_TF-IDF_ML.ipynb](Projects/02_SpamHam_using_TF-IDF_ML.ipynb)** - SMS spam classification using TF-IDF features
3. **[Projects/03_SpamHam_using_Word2Vec_ML.ipynb](Projects/03_SpamHam_using_Word2Vec_ML.ipynb)** - SMS spam classification using Word2Vec embeddings
4. **[Projects/04_Sentiment_Analysis_KindleReview.ipynb](Projects/04_Sentiment_Analysis_KindleReview.ipynb)** - Sentiment analysis on Kindle product reviews

### Datasets

The `Datasets/` folder contains real-world datasets used in projects:

- **kindleReview.csv** - Kindle product reviews with ratings for sentiment analysis
- **smsSpamCollection** - SMS messages labeled as spam or ham (legitimate) for classification

## 🎯 Learning Path

**Beginner:**

1. Start with `01_Tokenization_using_NLTK` to understand basic text processing
2. Progress through `02_Stemming` and `03_Lemmatization` for text normalization
3. Learn `04_Stopwords` for noise reduction

**Intermediate:** 4. Study `05_Parts_of_speech_Tagging` for linguistic features 5. Explore `06_Named_entity_recognition` for entity extraction 6. Master `07_Bag_of_words` and `08_TF-IDF` for text representation

**Advanced:** 7. Understand `09_Word2Vec_implementation` for word embeddings 8. Apply knowledge with spam classification projects (01-03) 9. Complete the sentiment analysis project (04)

## 📊 Project Highlights

### Spam Detection

Uses multiple feature extraction techniques (Bag of Words, TF-IDF, Word2Vec) with Naive Bayes classification to identify spam SMS messages. Compare results across different vectorization methods.

### Sentiment Analysis

Analyzes Kindle product reviews to determine positive/negative sentiment using preprocessing, feature extraction, and machine learning classification models.

## 🛠️ Technologies Used

- **NLTK** - Tokenization, stemming, lemmatization, POS tagging, NER
- **Scikit-learn** - Machine learning models and feature extraction (CountVectorizer, TfidfVectorizer)
- **Gensim/Word2Vec** - Word embeddings and semantic analysis
- **Pandas** - Data loading and manipulation
- **NumPy** - Numerical operations

## 🔑 Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required libraries:
  - `nltk` - Natural Language Toolkit
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning
  - `gensim` - Word2Vec implementation
  - `matplotlib`/`seaborn` - Visualization

## 📝 Notes

- Each notebook is self-contained and includes explanations, code examples, and outputs
- Datasets are included in the `Datasets/` folder for easy reproducibility
- Projects demonstrate the practical application of concepts learned in the core tutorials
- Code follows scikit-learn and pandas conventions

## 👨‍💻 Author

**Miskat Ahmmed**
- 📧 Email: memiskat362@gmail.com
- 🔗 GitHub: [@Miskat362](https://github.com/Miskat362)
- 📱 LinkedIn: [Connect](https://linkedin.com/in/miskat-ahmmed)


## 🤝 Contributing

Feel free to fork this repository and submit pull requests with improvements, additional notebooks, or new projects.

---

**Happy Learning!** Feel free to explore, experiment, and extend this repository with your own NLP projects.
