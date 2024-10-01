#!pip install networkx matplotlib
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.drawing.layout import bipartite_layout

ratings = pd.read_csv('1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python', encoding='latin1')
movies = pd.read_csv('1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin1')
users = pd.read_csv('1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], engine='python', encoding='latin1')

# Function to convert data types to native Python types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return obj.item().isoformat()
        else:
            return super(NumpyEncoder, self).default(obj)

# Merge the ratings with the movies data
merged_data = pd.merge(ratings, movies, left_on='item_id', right_on='movie_id')
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='s')
movies_dict = movies.set_index('movie_id').to_dict('index')
user_dict = {}

# Add user information to the user dictionary
for _, row in users.iterrows():
    user_id = row['user_id']
    user_info = {
        "user_id": user_id,
        "gender": row['gender'],
        "age": row['age'],
        "occupation": row['occupation'],
        "zip_code": row['zip_code'],
        "ratings": []
    }
    user_dict[user_id] = user_info

# Add movie ratings to the user dictionary
for _, row in ratings.iterrows():
    user_id = row['user_id']
    movie_id = row['item_id']
    movie_info = movies_dict[movie_id]
    rating_info = {
        'movie_id': movie_id,
        'rating': row['rating'],
        'timestamp': row['timestamp'],
        'title': movie_info['title'],
        'genres': movie_info['genres']
    }
    user_dict[user_id]['ratings'].append(rating_info)

# Save to JSON file using the custom NumpyEncoder
with open('restructured_user_data.json', 'w') as json_file:
    json.dump(user_dict, json_file, cls=NumpyEncoder, indent=4)

print("Restructured user data saved to 'restructured_user_data.json'")

# Load the combined user data from JSON file
with open('user_data.json', 'r') as json_file:
    user_data = json.load(json_file)

# Initialize the detailed graph
G_detailed = nx.Graph()

# Add users and their information to the detailed graph
for user_id, data in user_data.items():
    G_detailed.add_node(f'user_{user_id}', type='user', user_info=json.dumps(data['user_info']))
    for rating in data['ratings']:
        movie_id = rating['movie_id']
        movie_info = rating['movie_info']
        G_detailed.add_node(f'movie_{movie_id}', type='movie', movie_info=json.dumps(movie_info))
        G_detailed.add_edge(f'user_{user_id}', f'movie_{movie_id}', type='rated', rating=rating['rating'], timestamp=rating['timestamp'])
        
        # Add genre nodes and edges
        genres = movie_info['genres'].split('|')
        for genre in genres:
            if f'genre_{genre}' not in G_detailed:
                G_detailed.add_node(f'genre_{genre}', type='genre', name=genre)
            G_detailed.add_edge(f'movie_{movie_id}', f'genre_{genre}', type='belongs_to')

# Save the graph to a GraphML file
nx.write_graphml(G_detailed, 'knowledge_graph_detailed.graphml')

print("Graph saved to 'knowledge_graph_detailed.graphml'")