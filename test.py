
# CREDITS GENRES AND KEYWORDS BASED RECOMMENDER 
import pandas as pd 

# load keywords and credits 
credits = pd.read_csv('archive/credits.csv')
keywords = pd.read_csv('archive/keywords.csv')

# Load movies metadata
metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)

print(metadata)