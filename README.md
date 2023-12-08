# Allocated-Project_Movie-Recommendation-System


The system aims to analyze user preferences, historical interactions, and movie features to deliver relevant recommendations, ultimately increasing user retention and fostering a more enjoyable and tailored entertainment experience. The goal is to employ advanced recommendation algorithms that strike a balance between accuracy and diversity, ensuring users discover a broad range of movies aligned with their tastes while introducing them to potentially new and exciting content. Additionally, the system seeks to optimize recommendation performance metrics, such as precision, recall, and user satisfaction, through continuous evaluation and refinement of the recommendation models.

The dataset employed for this project is derived from Kaggle - https://www.kaggle.com/datasets/bandikarthik/movie-recommendation-system/data. It encompasses details such as:

User Interactions: User ratings, watch history, and preferences regarding documentaries.
Documentary Features: Details like genre, release date, director, and relevant topics.
Temporal Information: Timestamps for user interactions, aiding in understanding viewing patterns.
User-Specific Data: Insights into user-specific details, such as account creation date and viewing habits.

## Methodology
The dataset is thoughtfully divided into training and testing sets to ensure a comprehensive evaluation of the Movie Recommendation System's performance.

## Data Cleaning
Comprehensive data cleaning procedures are applied to address missing values and anomalies, ensuring the dataset is pristine and reliable for subsequent analysis

## Exploratory Data Analysis
In-depth exploratory data analysis techniques are employed to gain valuable insights into user interactions with movies. Visualization tools unravel patterns, correlations, and anomalies in key features, contributing to a deeper understanding of the data.

## Feature Engineering
To optimize recommendation performance, feature engineering strategies are implemented. This involves creating new features, transforming existing ones, or extracting meaningful information to enrich the dataset.

## Feature Scaling
Certain recommendation models benefit from feature scaling for optimal performance. Techniques like scaling and normalization are applied to ensure consistency and effectiveness in the modeling process.

## Data Imbalance
To mitigate potential class imbalance, the SMOTE (Synthetic Minority Oversampling Technique) library was employed. This technique synthetically increased the representation of the minority class ('fraudulent transactions').

## Models Training
State-of-the-art recommendation models are employed, and rigorous training and evaluation processes are conducted to select the most effective model. Performance metrics include accuracy, precision, recall, and user satisfaction, ensuring the system delivers high-quality personalized movie recommendations.


```python
# Assuming 'df' is a pandas DataFrame with columns 'user_id', 'movie_id', and 'rating'

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

# Define the Reader object
reader = Reader(rating_scale=(1, 5))

# Load the dataset
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use Singular Value Decomposition (SVD) algorithm
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

Make sure to install the Surprise library first using:
pip install scikit-surprise
