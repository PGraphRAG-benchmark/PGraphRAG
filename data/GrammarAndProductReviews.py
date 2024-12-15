import sys
import subprocess
import pkg_resources
import datasets
import pandas as pd
import random
from datasets import load_dataset
from collections import defaultdict
import tqdm
from datetime import datetime, timedelta
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
def install(package):
    try:
        pkg_resources.get_distribution(package)
        print(f"{package} is already installed.")
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed.")


install('pandas')
# Specify the path to your CSV file

file_path = '/GrammarandProductReviews.csv'

import pandas as pd

def load_and_clean_csv(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Replace empty strings with NaN
    data.replace('', pd.NA, inplace=True)
    # Drop rows where specified columns have missing values
    cleaned_data = data.dropna(subset=['id', 'reviews.title', 'reviews.text', 'reviews.username'])
    return cleaned_data
cleaned_data = load_and_clean_csv(file_path)
num_users_train = 20000
num_users_dev = 2500
num_users_test = 2500

from collections import defaultdict


# Transforming cleaned_data into a dictionary of user_ids to list of reviews
user_reviews_dict = defaultdict(list)
for index, row in cleaned_data.iterrows():
    user_reviews_dict[row['reviews.username']].append(row.to_dict())

# Convert dictionary to list of users (as expected by the existing code)
users_list = list(user_reviews_dict.items())

# Function to build global product counts
def build_global_product_counts(users):
    product_count = defaultdict(list)  # Store list of users who reviewed each product
    for user_id, reviews in users:
        # Focus on the first review in each user's profile
        first_review = reviews[0]
        product_count[first_review['id']].append(user_id)  # Track which users reviewed the first product
    return product_count

# Global product counts for the entire dataset
global_product_counts = build_global_product_counts(users_list)


def set_random_first_review_with_global_neighbor(user_reviews, global_product_counts):
    # Filter reviews that have at least one global neighbor
    valid_reviews = [review for review in user_reviews if len(global_product_counts[review['id']]) > 1]

    if valid_reviews:
        random_review = random.choice(valid_reviews)  # Pick a random review with global neighbors
        user_reviews.remove(random_review)
        user_reviews.insert(0, random_review)  # Set this random review as the first one
    else:
        return None  # No reviews with global neighbors, exclude the user

    return user_reviews
def add_user_to_split(reviewer_id, user_reviews, split_users, local_product_counts, added_users):
    # Add the reviewer to the split
    split_users.append((reviewer_id, user_reviews))
    added_users.add(reviewer_id)
    # Update local product counts for the split
    for review in user_reviews:
        local_product_counts[review['id']] += 1

def distribute_global_neighbors(users, num_users_train, num_users_dev, num_users_test, global_product_counts):
    train_users = []
    dev_users = []
    test_users = []

    # Initialize local product counts for train, dev, and test splits
    local_product_counts_train = defaultdict(int)
    local_product_counts_dev = defaultdict(int)
    local_product_counts_test = defaultdict(int)

    # Track which reviewers have already been added
    added_reviewers = set()

    # Exclude products with only one review
    filtered_product_list = [product_id for product_id, reviewer_ids in global_product_counts.items() if len(reviewer_ids) > 1]

    # Set the seed before shuffling to ensure reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(filtered_product_list)

    # Go through each product's global neighbors in a random order
    for product_id in filtered_product_list:
        reviewer_ids = global_product_counts[product_id]

        # Set the seed before shuffling reviewers to ensure reproducibility
        random.seed(RANDOM_SEED)
        random.shuffle(reviewer_ids)

        # Add the first two reviewers to the training set
        if len(train_users) < num_users_train - 1:
            for reviewer_id in reviewer_ids[:2]:  # Add the first two reviewers to the training set
                if reviewer_id in added_reviewers:
                    continue  # Skip reviewers who have already been added

                # Get the reviewer's reviews
                user_reviews = [user for user in users if user[0] == reviewer_id][0][1]

                # Add the reviewer to the training split
                add_user_to_split(reviewer_id, user_reviews, train_users, local_product_counts_train, added_reviewers)

        # Add the next two reviewers to the dev set if training set is filled
        elif len(dev_users) < num_users_dev - 1:
            for reviewer_id in reviewer_ids[:2]:  # Add the next two reviewers to the dev set
                if reviewer_id in added_reviewers:
                    continue  # Skip reviewers who have already been added

                # Get the reviewer's reviews
                user_reviews = [user for user in users if user[0] == reviewer_id][0][1]

                # Add the reviewer to the dev split
                add_user_to_split(reviewer_id, user_reviews, dev_users, local_product_counts_dev, added_reviewers)

        # Add the next two reviewers to the test set if dev set is filled
        elif len(test_users) < num_users_test - 1:
            for reviewer_id in reviewer_ids[:2]:  # Add the next two reviewers to the test set
                if reviewer_id in added_reviewers:
                    continue  # Skip reviewers who have already been added

                # Get the reviewer's reviews
                user_reviews = [user for user in users if user[0] == reviewer_id][0][1]

                # Add the reviewer to the test split
                add_user_to_split(reviewer_id, user_reviews, test_users, local_product_counts_test, added_reviewers)

        # Add subsequent reviewers for this product, ensuring they have local neighbors
        for i, reviewer_id in enumerate(reviewer_ids[2:], start=2):
            if reviewer_id in added_reviewers:
                continue  # Skip reviewers who have already been added

            # Get the reviewer's reviews
            user_reviews = [user for user in users if user[0] == reviewer_id][0][1]

            # Ensure subsequent reviewers have local neighbors in the train split
            if len(train_users) < num_users_train:
                if local_product_counts_train[product_id] > 0:  # Ensure local neighbor exists
                    add_user_to_split(reviewer_id, user_reviews, train_users, local_product_counts_train, added_reviewers)

            # Ensure subsequent reviewers have local neighbors in the dev split
            elif len(dev_users) < num_users_dev:
                if local_product_counts_dev[product_id] > 0:  # Ensure local neighbor exists
                    add_user_to_split(reviewer_id, user_reviews, dev_users, local_product_counts_dev, added_reviewers)

            # Ensure subsequent reviewers have local neighbors in the test split
            elif len(test_users) < num_users_test:
                if local_product_counts_test[product_id] > 0:  # Ensure local neighbor exists
                    add_user_to_split(reviewer_id, user_reviews, test_users, local_product_counts_test, added_reviewers)

            # Stop if all splits are full
            if len(train_users) >= num_users_train and len(dev_users) >= num_users_dev and len(test_users) >= num_users_test:
                break

    return train_users, dev_users, test_users, local_product_counts_train, local_product_counts_dev, local_product_counts_test
train_users, dev_users, test_users, local_product_counts_train, local_product_counts_dev, local_product_counts_test = distribute_global_neighbors(
    users_list, num_users_train, num_users_dev, num_users_test, global_product_counts
)

def get_local_neighbor_count(reviewers, local_product_counts):
    reviewer_neighbor_counts = []

    for reviewer_id, reviews in reviewers:
        first_review = reviews[0]  # Focus on the first review in the profile
        product_id = first_review['id']
        local_neighbors = local_product_counts[product_id] - 1  # Subtract 1 to exclude the current review
        reviewer_neighbor_counts.append((reviewer_id, local_neighbors))

    return pd.Series([count for _, count in reviewer_neighbor_counts]).value_counts().sort_index()
def print_local_neighbor_distribution_with_percentages(local_neighbor_count, split_name):
    total_reviewers = local_neighbor_count.sum()  # Total number of reviewers

    # Print the distribution with percentages
    print(f"{split_name} Local Neighbor Count Distribution (for the first review):")
    for neighbor_count, count in local_neighbor_count.items():
        percentage = (count / total_reviewers) * 100
        print(f"Reviewers with {neighbor_count} neighbors: {count} ({percentage:.2f}%)")
    print("\n")

def print_profile_size_distribution_with_percentages(profile_size_distribution, split_name):
    total_reviewers = profile_size_distribution.sum()  # Total number of reviewers

    # Print the distribution with percentages
    print(f"{split_name} Profile Size Distribution:")
    for profile_size, count in profile_size_distribution.items():
        percentage = (count / total_reviewers) * 100
        print(f"Reviewers with profile size {profile_size}: {count} ({percentage:.2f}%)")
    print("\n")
def get_profile_size_distribution(users):
    profile_sizes = [len(reviews) for reviewer_id, reviews in users]
    return pd.Series(profile_sizes).value_counts().sort_index()

# Collect local neighbor counts and profile sizes for train, dev, and test splits
train_local_neighbor_count = get_local_neighbor_count(train_users, local_product_counts_train)
dev_local_neighbor_count = get_local_neighbor_count(dev_users, local_product_counts_dev)
test_local_neighbor_count = get_local_neighbor_count(test_users, local_product_counts_test)

# Print local neighbor count distributions with percentages for each split
print_local_neighbor_distribution_with_percentages(train_local_neighbor_count, "Train")
print_local_neighbor_distribution_with_percentages(dev_local_neighbor_count, "Dev")
print_local_neighbor_distribution_with_percentages(test_local_neighbor_count, "Test")

import json
def process_user_data(users):
    out = []
    predictions = []
    for user_id, reviews in tqdm.tqdm(users):
        # Generate user input and output (as done before)
        user_entry = {"id": user_id}
        pred_entry = {
            "id": user_id,
            "title": reviews[0]['reviews.title'],  # Accessing the title of the first review
            "text": reviews[0]['reviews.text']     # Accessing the text of the first review
        }

        # Add full user profile to user_entry
        user_entry["profile"] = [
            {
                # 'rating': d['rating'],

                # 'text': d['text'],
                # Uncomment the lines below if you want to include these fields
                # 'timestamp': d['timestamp'],
                # 'helpful_vote': d['helpful_vote'],
                # 'verified_purchase': d['verified_purchase'],
                #  "gmap_id": d['gmap_id']
                'review_title': d['reviews.title'],
                 'review_text': d['reviews.text'],
                 'overall_rating': d['reviews.rating'],
                # 'reviewer_id': d['reviews.username'],
                # 'reviewer_name': d['reviews.username'],
                "product_id": d['id']
            }
            for d in reviews
        ]

        # Append entries to output lists
        out.append(user_entry)
        predictions.append(pred_entry)
    return out, predictions

# Logging info
logging.info("Processing train, test, and dev data")

# Process the train, test, and dev sets
out_train, predictions_train = process_user_data(train_users)
out_test, predictions_test = process_user_data(test_users)
out_dev, predictions_dev = process_user_data(dev_users)

# File name base (you can modify this as needed)
file_name_base = "GrammarAndProductreviewdataset"

# Function to write data to files
def write_to_file(data, file_suffix):
    file_name = f"{file_name_base}_{file_suffix}.json"
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

# Logging and writing the files
logging.info("Writing data to files")
write_to_file(out_train, "questions_train")
write_to_file(predictions_train, "outputs_train")
write_to_file(out_test, "questions_test")
write_to_file(predictions_test, "outputs_test")
write_to_file(out_dev, "questions_dev")
write_to_file(predictions_dev, "outputs_dev")

# Log completion
logging.info("Data successfully written to files.")
