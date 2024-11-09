# MovieLens Collaborative Filtering System

## Academic Context
This project was developed as part of a coursework assignment focusing on implementing a user-based collaborative filtering approach and exploiting it for producing group recommendations. The implementation demonstrates understanding of collaborative filtering algorithms, similarity metrics, and group recommendation strategies using the MovieLens 100K dataset.

## Project Overview
The project implements an advanced user-based collaborative filtering system with multiple similarity metrics, individual and group recommendation strategies, and a novel disagreement-aware recommendation approach with visualization capabilities.

## Features
- **Data Processing**
  - Efficient loading and preprocessing of MovieLens 100K dataset
  - User-movie rating matrix creation
  - Genre information processing

- **Similarity Metrics**
  - Pearson correlation-based similarity
  - Custom hybrid similarity combining:
    - Pearson correlation (40% weight)
    - Genre preference similarity (30% weight)
    - Rating pattern similarity (30% weight)

- **Recommendation Systems**
  - Individual user recommendations
  - Group recommendation strategies:
    - Average method
    - Least misery method
    - Novel disagreement-aware method

- **Analysis Tools**
  - Group disagreement visualization
  - Rating prediction analysis
  - Comprehensive movie information display

## Prerequisites
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict
```

## Dataset
The project uses the MovieLens 100K dataset, specifically:
- `u.data`: Contains user ratings
- `u.item`: Contains movie information

## Setup and Installation
1. Install required Python packages:
```bash
pip install pandas numpy scipy matplotlib
```

2. Download the MovieLens 100K dataset and ensure the following files are in your working directory:
   - `u.data` (rating data)
   - `u.item` (movie information)

## Usage
Basic usage example:
```python
# Initialize recommender
recommender = MovieLens100kRecommender()
recommender.load_data()

# Get individual recommendations
user_recommendations = recommender.get_recommendations(
    user_id=1,
    n=5,
    use_custom_similarity=True
)

# Get group recommendations
group_recommendations = recommender.get_group_recommendations(
    group_users=[1, 2, 3, 4],
    n=5,
    method='disagreement_aware'
)

# Analyze group disagreements
recommender.analyze_group_disagreements(group_users=[1, 2, 3, 4])
```

## Implementation Details

### Custom Similarity Metric
The custom similarity metric combines three components:
1. **Pearson Correlation**: Measures rating pattern correlation
2. **Genre Preference**: Compares users' genre preferences based on their rated movies
3. **Rating Pattern**: Analyzes the distribution of rating values

### Group Recommendation Methods
- **Average**: Simple averaging of predicted ratings
- **Least Misery**: Minimum predicted rating approach
- **Disagreement-Aware**: Novel approach that penalizes recommendations with high standard deviation in predicted ratings

### Disagreement Analysis
The system includes visualization tools to analyze group disagreements:
- Bar charts showing standard deviation of predictions
- Movie-wise disagreement analysis
- Customizable number of movies for analysis

## Project Structure
```
├── movielens_recommender.py  # Main implementation file
├── data/
│   ├── u.data               # Rating data
│   └── u.item               # Movie information
└── README.md
```

## Performance
- Efficiently handles 100,000 ratings from 943 users on 1,682 movies
- Matrix operations for similarity calculations
- Customizable k-nearest neighbors for prediction

## Future Improvements
Potential areas for enhancement:
- Implementation of matrix factorization techniques
- Additional similarity metrics
- More group recommendation strategies
- Performance optimization for larger datasets

## Acknowledgments
- MovieLens dataset provided by GroupLens Research
- Based on collaborative filtering research literature