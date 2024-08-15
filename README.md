
# Content-Based Recommendation System
This System was an initiative for Abhyuday, IIT Bombay's Career Counselling Campaign.
This repository contains the code for a content-based recommendation system using Word2Vec embeddings and a hybrid K-NN and cosine similarity model. 

## Overview
This project implements a content-based recommendation system that uses CBoW Word2Vec embeddings to generate relevant tags for careers. The system provides career recommendations by applying a hybrid K-NN and cosine similarity model. The model was deployed using Flask.

## Features
- **Content-based filtering:** Recommends careers based on similarity to user input.
- **Word2Vec embeddings:** Uses CBoW Word2Vec embeddings to create meaningful representations of career tags.
- **Hybrid model:** Combines K-NN and cosine similarity for accurate recommendations.
- **Flask-based web app:** Simple and user-friendly web interface for testing recommendations.

## Dataset
The original dataset is proprietary to Abhyuday. It consists of data from psychometric tests of over 300 students and tags for career profiles.

## Model
The model uses a hybrid approach combining K-NN and cosine similarity:
- **Word2Vec Model:** Trained on career tags to create embeddings.
- **K-NN Model:** Identifies the nearest neighbors based on cosine similarity.
- **Cosine Similarity:** Further refines the recommendations by ranking them based on similarity scores.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SharvariMedhe/content-based-recommendation-system.git
   cd content-based-recommendation-system
