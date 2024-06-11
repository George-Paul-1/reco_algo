# We are going to compute pairwise cosine similarity score based off movie 
# descriptions, recommending films based off of how similar their descriptions
# are. 

import pandas as pd 
# Load movies metadata
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
print('\n')
print(metadata['overview'].head() + '\n')
print('\n')

# Printing first five rows of descriptions in dataset

# The problem at hand is a natural language processing provlem. Hence you need to 
# exract some kind of features from the above text data before you can compute 
# similarities and 0r dissimilarities between them. 

# To put it simply, it is not possible to compute the similarity between any two 
# overviews in their raw forms. To do this, you need to compute the word vectors of each 
# overview or document, as it will be called from now on.

# As the name suggests, word vectors are vectorized representation of words in 
# a document. The vectors carry a semantic meaning with it. For example, man & 
# king will have vector representations close to each other while man & woman would 
# have representation far from each other.

# You will compute Term Frequency-Inverse Document Frequency 
# (TF-IDF) vectors for each document. This will give you a matrix where each 
# column represents a word in the overview vocabulary (all the words that appear 
# in at least one document), and each column represents a movie, as before.

# We can use scikit-learn's tfIdfVectorizer class to produce a TF-IDF matrix in a couple 
# of lines 

from sklearn.feature_extraction.text import TfidfVectorizer
#Import TfIdfVectorizer from scikit-learn

# define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words = 'english') 

# Replace NaN with an empty string

metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix 
print('TF-IDF Matrix Dimensions')
print(tfidf_matrix.shape)
print('\n')
print('This means that 70,827 different vocabularies or words in dataset have 45,000 movies')
print('\n')
# Array mapping from feature integer indices to feature name.
print('Slice of the vocab features we now have')
print(tfidf.get_feature_names_out()[5000:5010])
print('\n')

# With this matrix in hand, you can now compute a similarity score. 
# There are several similarity metrics that you can use for this, such as the 
# manhattan, euclidean, the Pearson, and the cosine similarity scores. Again, there is no 
# right answer to which score is the best. Different scores work well in different 
# scenarios, and it is often a good idea to experiment with different metrics and observe the results.

# You will be using the cosine similarity to calculate a numeric quantity that denotes 
# the similarity between two movies. You use the cosine similarity score since it is independent of 
# magnitude and is relatively easy and fast to calculate (especially when used in conjunction with TF-IDF 
# scores, which will be explained later).


# Since you have used the TF-IDF vectorizer, calculating the dot product between each vector will directly give 
# you the cosine similarity score. Therefore, you will use sklearn's linear_kernel() instead of cosine_similarities() 
# since it is faster.

# This would return a matrix of shape 45466x45466, which means each movie overview cosine similarity score with 
# every other movie overview. Hence, each movie will be a 1x45466 column vector where each column will be a similarity 
# score with each movie.

# importing linear_kernel
from sklearn.metrics.pairwise import linear_kernel 

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print('shape of the cosine_similarity matrix')
print(cosine_sim.shape)
print('\n')
print('first element in matrix')
print(cosine_sim[1])
print('\n')

# Construct a reverse map of indices and movie titles 
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
print('we now construct a reverse map of indices and movie titles, here is a slice of it')
print(indices[:10])
print('\n')
# This gives us a sorted table that we can use to get the index position of each element in the dataset 
# from its title

# You are now in good shape to define your recommendation function. These are the following steps you'll follow:

# 1. Get the index of the movie given its title.

# 2. Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list 
# of tuples where the first element is its position, and the second is the similarity score.

# 3. Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.

# 4. Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar 
# to a particular movie is the movie itself).

# 5. Return the titles corresponding to the indices of the top elements.

def get_recommendations(title, cosine_sim=cosine_sim): 
    # Get the index of the movie that matches the title
    idx = indices[title]

# Get the pairwise similarity scores of all movies with that movie 
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices 
    movie_indices = [i[0] for i in sim_scores] 

    # Return the top 10 most similar movies 
    return metadata['title'].iloc[movie_indices]

print('Getting recommendations similar to the Dark Knight Rises: ')
print(get_recommendations('The Dark Knight Rises'))
print('\n')

print('getting recommendations for jumanji: ')
print(get_recommendations('Jumanji'))
print('\n')

print('Getting recommendations for The Godfather: ')
print(get_recommendations('The Godfather'))
print('\n')

print(get_recommendations('Toy Story'))


# CREDITS GENRES AND KEYWORDS BASED RECOMMENDER 

