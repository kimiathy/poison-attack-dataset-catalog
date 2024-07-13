# Poison Attack Datasets Catalog

## Overview
This repository provides a catalog of datasets commonly used in data and model poisoning research. Each dataset entry includes metadata, a list of relevant papers, and preprocessing instructions.

## Table of Contents
- [Datasets](#datasets)
  - [MNIST](#mnist)
  - [CIFAR-10 and CIFAR-100](#cifar-10-and-cifar-100)
  - [20 Newsgroups](#20-newsgroups)
  - [AG's News Topic Classification](#ags-news-topic-classification)
  - [CheXpert](#chexpert)
  - [Enron Email Dataset](#enron-email-dataset)
  - [IMDB Movie Reviews](#imdb-movie-reviews)
  - [ImageNet](#imagenet)
  - [KDD Cup 1999 Data](#kdd-cup-1999-data)
  - [Labeled Faces in the Wild (LFW)](#labeled-faces-in-the-wild-lfw)
  - [MIMIC-III](#mimic-iii)
  - [PhysioNet](#physionet)
  - [Reuters-21578 Text Categorization Collection](#reuters-21578-text-categorization-collection)
  - [SpamBase Dataset](#spambase-dataset)
  - [Stanford Sentiment Treebank (SST)](#stanford-sentiment-treebank-sst)
  - [The Cancer Imaging Archive (TCIA)](#the-cancer-imaging-archive-tcia)
  - [TIMIT Acoustic-Phonetic Continuous Speech Corpus](#timit-acoustic-phonetic-continuous-speech-corpus)
  - [Twitter Sentiment Analysis Dataset](#twitter-sentiment-analysis-dataset)
  - [UCI Machine Learning Repository](#uci-machine-learning-repository)
  - [Yelp Review Dataset](#yelp-review-dataset)
  - [YouTube Faces Dataset](#youtube-faces-dataset)
  - [iDASH](#idash)

## Datasets

### MNIST
- **Description**: Handwritten digits dataset for machine learning.
- **Size**: 53 MB
- **Classes**: 10 (digits 0-9)
- **Samples**: 70,000 (60,000 training, 10,000 testing)
- **Format**: IDX (binary), 28x28 grayscale images
- **Preprocessing**: Normalize pixel values to 0-1.
- **Papers**: 
  - [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
  - [Additional Papers](docs/MNIST_papers.md)

### CIFAR-10 and CIFAR-100
- **Description**: Color image datasets for object recognition.
- **Size**: CIFAR-10: 170 MB, CIFAR-100: 161 MB
- **Classes**: CIFAR-10: 10, CIFAR-100: 100
- **Samples**: 60,000 each
- **Format**: tar.gz, binary version for C, Python, Matlab
- **Preprocessing**: Normalize pixel values, one-hot encode labels.
- **Papers**: 
  - [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
  - [Additional Papers](docs/CIFAR_papers.md)

### 20 Newsgroups
- **Description**: Text dataset for text classification and clustering.
- **Size**: 14 MB
- **Classes**: 20 newsgroups
- **Samples**: 20,000 documents
- **Format**: ASCII text
- **Preprocessing**: Text cleaning, tokenization, stop words removal.
- **Papers**: 
  - [Mitigating Backdoor Attacks in LSTM-based Text Classification Systems](https://www-sciencedirect-com.ledproxy2.uwindsor.ca/science/article/pii/S0925231221006639?via%3Dihub)

### [AG's News Topic Classification](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Description**: News articles from various sources for text classification.
- **Size**: 29 MB
- **Classes**: 4 (World, Sports, Business, Science/Technology)
- **Samples**: 120,000 (training), 7,600 (testing)
- **Format**: CSV
- **Preprocessing**: Text cleaning, tokenization, one-hot encoding.
- **Papers**: Not listed.

### CheXpert
- **Description**: Chest X-rays for medical image analysis.
- **Size**: 439 GB
- **Classes**: 14 pathologies
- **Samples**: 224,316 images
- **Format**: DICOM
- **Preprocessing**: Resizing, normalization, handling uncertain labels.
- **Papers**: 
  - [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031)
  - [Suppressing Poisoning Attacks on Federated Learning for Medical Imaging](https://link-springer-com.ledproxy2.uwindsor.ca/chapter/10.1007/978-3-031-16452-1_64)

### [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)
- **Description**: Emails from Enron's senior management.
- **Size**: 1.7 GB
- **Classes**: Not categorized.
- **Samples**: 500,000 emails
- **Format**: Text files
- **Preprocessing**: Parsing, cleaning, vectorization.
- **Papers**: 
  - [Stronger Data Poisoning Attacks Break Data Sanitization Defenses](https://link-springer-com.ledproxy2.uwindsor.ca/article/10.1007/s10994-021-06119-y)

### [IMDB Movie Reviews](https://keras.io/api/datasets/imdb/)
- **Description**: Movie reviews for sentiment analysis.
- **Size**: 80 MB
- **Classes**: 2 (Positive, Negative)
- **Samples**: 50,000 reviews
- **Format**: Text files
- **Preprocessing**: Text cleaning, tokenization, vectorization.
- **Papers**: 
  - [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/)

### [ImageNet](http://www.image-net.org/)
- **Description**: Large-scale image dataset for object detection and classification.
- **Size**: 150 GB
- **Classes**: 20,000
- **Samples**: 14 million images
- **Format**: JPEG
- **Preprocessing**: Resizing, normalization, augmentation.
- **Papers**: 
  - [Black-Box Dataset Ownership Verification via Backdoor Watermarking](https://ieeexplore-ieee-org.ledproxy2.uwindsor.ca/document/10097580)

### [KDD Cup 1999 Data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Description**: Network intrusion detection dataset.
- **Size**: 740 MB
- **Classes**: 23 (attack types)
- **Samples**: 5 million records
- **Format**: CSV
- **Preprocessing**: Data cleaning, normalization, encoding.
- **Papers**: 
  - [The 1998 DARPA Intrusion Detection Evaluation](https://apps.dtic.mil/sti/pdfs/ADA378445.pdf)

### [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- **Description**: Images of faces for face recognition.
- **Size**: 233 MB
- **Classes**: 5,749 individuals
- **Samples**: 13,233 images
- **Format**: JPEG
- **Preprocessing**: Face detection, alignment, normalization.
- **Papers**: 
  - [DeepPoison: Feature Transfer Based Stealthy Poisoning Attack for DNNs](https://ieeexplore-ieee-org.ledproxy2.uwindsor.ca/document/9359658)

### [MIMIC-III](https://mimic.mit.edu/)
- **Description**: Critical care health data.
- **Size**: 35 GB
- **Classes**: Multiple tables
- **Samples**: 40,000 patients
- **Format**: CSV
- **Preprocessing**: Handling missing values, data cleaning, transformation.
- **Papers**: 
  - [MIMIC-III, a Freely Accessible Critical Care Database](https://www.nature.com/articles/sdata201635)

### [PhysioNet](https://physionet.org/)
- **Description**: Physiological signal datasets.
- **Size**: Varies
- **Classes**: Varies
- **Samples**: Varies
- **Format**: WFDB
- **Preprocessing**: Noise reduction, feature extraction, normalization.
- **Papers**: 
  - [PhysioBank, PhysioToolkit, and PhysioNet](https://www.ahajournals.org/doi/10.1161/01.CIR.101.23.e215)

### [Reuters-21578 Text Categorization Collection](https://archive.ics.uci.edu/datasets)
- **Description**: News documents for text categorization.
- **Size**: 43 MB
- **Classes**: 90 (commonly used: 10 or 20)
- **Samples**: 21,578 documents
- **Format**: SGML
- **Preprocessing**: Parsing, tokenization, vectorization.
- **Papers**: 
  - [Text Categorization with Support Vector Machines](https://link.springer.com/chapter/10.1007/BFb0026683)

### [SpamBase Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data)
- **Description**: Spam email dataset for spam detection.
- **Size**: 0.5 MB
- **Classes**: 2 (Spam, Not-Spam)
- **Samples**: 4,601 instances
- **Format**: CSV
- **Preprocessing**: Normalization, splitting into training and testing sets.
- **Papers**: 
  - [Label Flipping Attacks Against Naive Bayes on Spam Filtering Systems](https://link-springer-com.ledproxy2.uwindsor.ca/article/10.1007/s10489-020-02086-4)

### [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html)
- **Description**: Sentiment analysis on movie reviews.
- **Size**: Small
- **Classes**: 5 sentiment classes
- **Samples**: 215,154 phrases
- **Format**: CSV
- **Preprocessing**: Text cleaning, tokenization, vectorization.
- **Papers**: 
  - [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

### [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **Description**: Medical images for cancer research.
- **Size**: Varies
- **Classes**: Various cancer types
- **Samples**: Thousands of patients
- **Format**: DICOM
- **Preprocessing**: DICOM file decoding, image normalization.
- **Papers**: 
  - [The Cancer Imaging Archive (TCIA)](https://link.springer.com/article/10.1007/s10278-013-9622-7)

### [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1)
- **Description**: Continuous speech corpus.
- **Size**: 440 MB
- **Classes**: Speech from 8 dialect regions
- **Samples**: 6,300 sentences
- **Format**: NIST SPHERE
- **Preprocessing**: Feature extraction, transcription parsing.
- **Papers**: 
  - [A Universal Identity Backdoor Attack Against Speaker Verification](https://www.isca-speech.org/archive/interspeech_2022/zhao22e_interspeech.html)

### [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description**: Sentiment analysis on tweets.
- **Size**: Hundreds of MB
- **Classes**: 2 (Positive, Negative)
- **Samples**: 1,578,627 tweets
- **Format**: CSV
- **Preprocessing**: Text cleaning, tokenization, vectorization.
- **Papers**: 
  - [Man vs. Machine: Practical Adversarial Detection of Malicious Crowdsourcing Workers](https://www-webofscience-com.ledproxy2.uwindsor.ca/wos/woscc/full-record/WOS:000495357000016)

### [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
- **Description**: Collection of various datasets for ML tasks.
- **Size**: Varies
- **Classes**: Varies
- **Samples**: Varies
- **Format**: CSV, ARFF
- **Preprocessing**: Varies per dataset.
- **Papers**: 
  - [Jangseung: A Guardian for ML Algorithms to Protect Against Poisoning Attacks](https://ieeexplore-ieee-org.ledproxy2.uwindsor.ca/document/9562816)

### [Yelp Review Dataset](https://www.yelp.com/dataset)
- **Description**: Reviews, ratings, and business data.
- **Size**: 10 GB
- **Classes**: Varies
- **Samples**: Several million reviews
- **Format**: JSON
- **Preprocessing**: Text cleaning, tokenization, encoding.
- **Papers**: Not listed.

### [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/)
- **Description**: Videos of faces for recognition tasks.
- **Size**: 186 GB
- **Classes**: 1,595 individuals
- **Samples**: 3,425 videos
- **Format**: AVI
- **Preprocessing**: Face detection, alignment, frame extraction.
- **Papers**: 
  - [DeepPoison: Feature Transfer Based Stealthy Poisoning Attack for DNNs](https://ieeexplore-ieee-org.ledproxy2.uwindsor.ca/document/9359658)

### [iDASH](https://idash.ucsd.edu/)
- **Description**: Healthcare data and algorithms for privacy-preserving research.
- **Size**: Varies
- **Classes**: Varies
- **Samples**: Varies
- **Format**: Varies
- **Preprocessing**: Varies per dataset.
- **Papers**: Not listed.
