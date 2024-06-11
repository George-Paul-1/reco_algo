import pandas as pd 
# Load movies metadata
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
# Print the first three rows 
# print(metadata.head(3))

# \begin{equation} \text Weighted Rating (\bf WR) = \left({{\bf v} \over {\bf v} + {\bf m}} \cdot R\right) + \left({{\bf m} \over {\bf v} + {\bf m}} \cdot C\right) \end{equation}

# In the above equation,

# v is the number of votes for the movie;

# m is the minimum votes required to be listed in the chart;

# R is the average rating of the movie;

# C is the mean vote across the whole report.

print('\n')

# Calculate mean of votes average column 
print('AVERAGE CORE OF FILM IN DATASET')
C = metadata['vote_average'].mean()
print(C)
print('\n')
# Above is the average rating of a movie on imdb (roughly 5.6 on a scale of 10) 

# Calculate the minimum number of votes required to be in chart m, 
# Pandas makes this simple with the quantile method 
print('MINIMUM NUMBER OF VOTES REQUIRED TO BE CONSIDERED')
m = metadata['vote_count'].quantile(0.90)
print(m)
print('\n')

# Here we're trying to calculate the cutoff 'm' that will simply remove the movies which 
# have a number below a certain threshold. Here we using 0.90 to represent the 90th percentile 
# i.e. the movies we're considering must have more votes than at least 90% of the movies on the list 


# Since now you have the m you can simply use a greater than equal to condition to filter 
# out movies having greater than equal to 160 vote counts:

# You can use the .copy() method to ensure that the new q_movies DataFrame created is independent of 
# your original metadata DataFrame. In other words, any changes made to the q_movies DataFrame 
# will not affect the original metadata data frame.

# Filter out all qualified movies into a new dataframe

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

print('FILTERING MOVIES THAT HAVE A VOTE COUNT GREATER THAN OR EQUAL TO 160, PRINTING THE NEW DIMENSIONS OF DATASET')
print(q_movies.shape)
print('\n')
print('ORIGINAL DIMENSIONS OF THE DATASET')
print(metadata.shape)
print('\n')

print('Roughly 10% of movies qualify to be in list') 
print('\n')

def weighted_rating(x, m=m, C=C): 
    v = x['vote_count']
    R = x['vote_average']
    # calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C) 

# Define a new feature 'score' and calculate its value with "weighted_rating"

q_movies['score'] = q_movies.apply(weighted_rating, axis=1) 

# Finally, lets sort the Dataframe in descending order based on the score feature column and output 
# the title, vote count, vote average and weighted rating(score) of the top 20
# movies. 

# Sort movies based on score calculated above 

q_movies = q_movies.sort_values('score', ascending=False) 

# Print the top 15 movies 
print('TOP 15 MOVIES AFTER CALC')
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))
print('\n')

