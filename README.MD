# 🎬 Hybrid Movie Recommendation System

A hybrid recommendation system that combines **content-based filtering** and **collaborative filtering** to generate personalized movie recommendations. The system is deployed using **FastAPI** for real-time API access.

---

## 🚀 Overview

This project builds a recommendation engine using:

* **Content-Based Filtering** → based on movie genres
* **Collaborative Filtering** → based on user ratings
* **Hybrid Model** → combines both approaches using weighted scoring

The goal is to create a system that provides **relevant and personalized recommendations** instead of generic suggestions.

---

## 🧠 How It Works

### 1. Content-Based Filtering

* Extracts movie genres
* Converts text → numerical vectors using `CountVectorizer`
* Computes similarity using cosine similarity

---

### 2. Collaborative Filtering

* Builds a **user-item matrix**
* Identifies similar users using cosine similarity
* Recommends movies liked by similar users

---

### 3. Hybrid Recommendation

Final score is computed as:

Score = α × Content Score + β × Collaborative Score

* Combines both signals
* Reduces bias of individual methods
* Produces more balanced recommendations

---

## 📊 Features

* Personalized recommendations per user
* Avoids recommending already watched movies
* Combines multiple recommendation strategies
* FastAPI-based backend
* Interactive testing via Swagger UI

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Uvicorn

---

## 📁 Project Structure

project/
│
├── app.py              # FastAPI application
├── movies.csv          # Movie dataset
├── ratings.csv         # User ratings dataset
├── requirements.txt    # Dependencies
└── README.md

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

---

### 2. Install Dependencies

pip install -r requirements.txt

---

### 3. Run the Server

uvicorn app:app --reload

---

### 4. Open API Docs

http://127.0.0.1:8000/docs

---

## 📡 API Usage

### Endpoint

GET /recommend

### Parameters

* user_id (int)
* movie (string)

---

### Example Request

/recommend?user_id=1&movie=Toy%20Story%20(1995)

---

### Example Response

{
"user_id": 1,
"movie": "Toy Story (1995)",
"recommendations": [
"Toy Story 2 (1999)",
"A Bug's Life (1998)"
]
}

---

## 📈 Evaluation

The system can be evaluated using **Precision@K**, which measures how many recommended movies are relevant based on user similarity.

---

## ⚠️ Limitations

* Cold start problem for new users and movies
* Limited to genre-based content features
* No deep learning or advanced embeddings used

---

## 🔥 Future Improvements

* Add matrix factorization (SVD)
* Use embeddings for better content understanding
* Improve cold-start handling
* Deploy with caching for faster performance
* Build frontend UI

---

## 📌 Key Learnings

* Designing hybrid recommendation systems
* Balancing multiple signals (content + collaborative)
* Converting ML logic into a deployable API
* Importance of evaluation and system design

---

## 📜 License

This project is open-source and intended for educational purposes.

## Author

Om Vaghani  