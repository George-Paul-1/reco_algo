
# CREDITS GENRES AND KEYWORDS BASED RECOMMENDER 
import pandas as pd 
print('\n')
print('Getting recommendations..')
print('\n')
# load keywords and credits 
credits = pd.read_csv('archive/credits.csv')
keywords = pd.read_csv('archive/keywords.csv')

# Load movies metadata
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
# print('\n')
# print(metadata['overview'].head() + '\n')
# print('\n')

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# convert IDs to int. Required for merging. 

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main dataframe 
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id') 

#print('merged datasets')

# Print the first two movies of your newly merged data
# print(metadata.head(2))

# From your new features, cast, crew, and keywords, you need to extract the three most important actors, the director and the keywords associated with that movie.

# But first things first, your data is present in the form of "stringified" lists. You need to convert them into a way that is usable for you.

# parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres'] 
for feature in features: 
    metadata[feature] = metadata[feature].apply(literal_eval) 

# This function evaluates a string containing a Python literal or container, such as a list or dictionary, and returns the corresponding Python object.
# By applying literal_eval to each feature column, the code is converting the stringified data into actual Python objects, which can be more easily manipulated and analyzed.

# Next, you write functions that will help you to extract the required information from each feature.

# First, you'll import the NumPy package to get access to its NaN constant. Next, you can use it to write the get_director() function:

# import numpy
import numpy as np

# get the directors name from the crew feature. If the director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Next, you will write a function that will return the top 3 elements or the entire list, whichever is more. Here the list refers to the cast, keywords, and genres.

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names 
    # return empty list in case of missing/malformed data 
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director) 
features = ['cast', 'keywords', 'genres']
for feature in features: 
    metadata[feature] = metadata[feature].apply(get_list) 

#print('modified dataset features..')

# print the new featurs of the first 3 films 
# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# The next step would be to convert the names and keyword instances into lowercase and strip 
# all the spaces between them

# Removing the spaces between words is an important preprocessing step. It is done 
# so that your vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" 
# as the same. After this processing step, teh aforementioned actors will be represented
# as johnnydepp and johnnygalecki and will be distint to you vectorizer

# Stripping whitespace from names and keywords 
#print('stripping whitespace and lowercasing names and keywords...')
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else: 
        # check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else: 
            return ''

# Apply clean_data function to your features 
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data) 

# Your are not in a position to create your 'metadata soup', which is a string that contains 
# all the metadata that you want to feed your vectorizer(names actors, director and keywords) 

# the create_soup function will simply join all the required columns by a space. This is the 
# final preprocessing step, and the output of this function will be fed into the word vector model 
#print('Creating soup...')
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


# Create a new soup feature 
metadata['soup'] = metadata.apply(create_soup, axis=1) 

#print(metadata[['soup']].head(3))

# The next steps are the same as what you did with your plot description based recommender. 
# One key difference is that you use the CountVectorizer() instead of TF-IDF. This is because 
# you do not want to down-weight the actor/director's presence if he or she has acted or 
# directed in relatively more movies. It doesn't make much intuitive sense to down-weight them 
# in this context.

# The major difference between CountVectorizer() and TF-IDF is the inverse document frequency (IDF) 
# component which is present in later and not in the former.

# Import countVectorizer and create the count matric 
from sklearn.feature_extraction.text import CountVectorizer


#print('Initialising count vectorizer...')
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
#print('count vectorizer initialized and matrix created')

vocabulary = count.vocabulary_ 
#print('size of vocabulary:', len(vocabulary))

#print('sample of terms from the vocab:')
sample_terms = list(vocabulary.keys())[:10]
#print(sample_terms)

#print('returning dimensions of count_matrix...')
#print(count_matrix.shape) 


# Compute the Cosine similarity matrix based on the count_matrix 
from sklearn.metrics.pairwise import cosine_similarity 

#print('Computing cosine similarity matrix...')
cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 

#print('Creating sorted list of indices...')
indices = pd.Series(metadata.index, index=metadata['title'])



def get_recommendations(title, cosine_sim=cosine_sim2): 
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

# TESTING OUT 

print('If you like The Dark Knight Rises...')
print(get_recommendations('The Dark Knight Rises', cosine_sim2))

print('\n')

