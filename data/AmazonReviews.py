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

def install(package):
    try:
        pkg_resources.get_distribution(package)
        print(f"{package} is already installed.")
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed.")


install('datasets')

# Rest of your script follows


logging.basicConfig(level=logging.INFO)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
ds = dataset["full"]

# Use a fixed cutoff date (e.g., January 1, 2024)
FIXED_CUTOFF_DATE = datetime(2024, 1, 1)
DATE_CUTOFF = FIXED_CUTOFF_DATE - timedelta(days=3*365)
def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp / 1000)

logging.info("Filtering dataset for time cutoff")
filtered_ds = [d for d in tqdm.tqdm(ds) if convert_timestamp(d['timestamp']) > DATE_CUTOFF]
logging.info("Organizing data by user_id")
d_dic = defaultdict(list)
for d in filtered_ds:
    d_dic[d["user_id"]].append(d)
num_users_train = 20000
num_users_dev = 2500
num_users_test = 2500

# Convert dictionary to list of users
users_list = list(d_dic.items())

# Build global product counts across the entire dataset
def build_global_product_counts(users):
    product_count = defaultdict(list)  # Store list of users who reviewed each product
    for user_id, reviews in users:
        # Focus on the first review in each user's profile
        first_review = reviews[0]
        product_count[first_review['asin']].append(user_id)  # Track which users reviewed the first product
    return product_count

# Global product counts for the entire dataset
global_product_counts = build_global_product_counts(users_list)
def set_random_first_review_with_global_neighbor(user_reviews, global_product_counts):
    # Filter reviews that have at least one globxal neighbor
    valid_reviews = [review for review in user_reviews if len(global_product_counts[review['asin']]) > 1]

    if valid_reviews:
        random_review = random.choice(valid_reviews)  # Pick a random review with global neighbors
        user_reviews.remove(random_review)
        user_reviews.insert(0, random_review)
    else:
        return None  # No reviews with global neighbors, exclude the user

    return user_reviews
def add_user_to_split(user_id, user_reviews, split_users, local_product_counts, added_users):
    # Add the user to the split
    split_users.append((user_id, user_reviews))
    added_users.add(user_id)
    # Update local product counts for the split
    for review in user_reviews:
        local_product_counts[review['asin']] += 1


def distribute_global_neighbors(users, num_users_train, num_users_dev, num_users_test, global_product_counts):
    train_users = []
    dev_users = []
    test_users = []

    # Initialize local product counts for train, dev, and test splits
    local_product_counts_train = defaultdict(int)
    local_product_counts_dev = defaultdict(int)
    local_product_counts_test = defaultdict(int)

    # Track which users have already been added
    added_users = set()

    # Exclude products with only one review
    filtered_product_list = [asin for asin, user_ids in global_product_counts.items() if len(user_ids) > 1]

    # Set the seed before shuffling to ensure reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(filtered_product_list)

    # Go through each product's global neighbors in a random order
    for asin in filtered_product_list:
        user_ids = global_product_counts[asin]

        # Set the seed before shuffling users to ensure reproducibility
        random.seed(RANDOM_SEED)
        random.shuffle(user_ids)

        # Add the first two users to the training set
        if len(train_users) < num_users_train - 1:
            for user_id in user_ids[:2]:  # Add the first two users to the training set
                if user_id in added_users:
                    continue  # Skip users who have already been added

                # Get the user's reviews
                user_reviews = [user for user in users if user[0] == user_id][0][1]

                # Add the user to the training split
                add_user_to_split(user_id, user_reviews, train_users, local_product_counts_train, added_users)

        # Add the next two users to the dev set if training set is filled
        elif len(dev_users) < num_users_dev - 1:
            for user_id in user_ids[:2]:  # Add the next two users to the dev set
                if user_id in added_users:
                    continue  # Skip users who have already been added

                # Get the user's reviews
                user_reviews = [user for user in users if user[0] == user_id][0][1]

                # Add the user to the dev split
                add_user_to_split(user_id, user_reviews, dev_users, local_product_counts_dev, added_users)

        # Add the next two users to the test set if dev set is filled
        elif len(test_users) < num_users_test - 1:
            for user_id in user_ids[:2]:  # Add the next two users to the test set
                if user_id in added_users:
                    continue  # Skip users who have already been added

                # Get the user's reviews
                user_reviews = [user for user in users if user[0] == user_id][0][1]

                # Add the user to the test split
                add_user_to_split(user_id, user_reviews, test_users, local_product_counts_test, added_users)

        # Add subsequent users for this product, ensuring they have local neighbors
        for i, user_id in enumerate(user_ids[2:], start=2):
            if user_id in added_users:
                continue  # Skip users who have already been added

            # Get the user's reviews
            user_reviews = [user for user in users if user[0] == user_id][0][1]

            # Ensure subsequent users have local neighbors in the train split
            if len(train_users) < num_users_train:
                if local_product_counts_train[asin] > 0:  # Ensure local neighbor exists
                    add_user_to_split(user_id, user_reviews, train_users, local_product_counts_train, added_users)

            # Ensure subsequent users have local neighbors in the dev split
            elif len(dev_users) < num_users_dev:
                if local_product_counts_dev[asin] > 0:  # Ensure local neighbor exists
                    add_user_to_split(user_id, user_reviews, dev_users, local_product_counts_dev, added_users)

            # Ensure subsequent users have local neighbors in the test split
            elif len(test_users) < num_users_test:
                if local_product_counts_test[asin] > 0:  # Ensure local neighbor exists
                    add_user_to_split(user_id, user_reviews, test_users, local_product_counts_test, added_users)

            # Stop if all splits are full
            if len(train_users) >= num_users_train and len(dev_users) >= num_users_dev and len(test_users) >= num_users_test:
                break

    return train_users, dev_users, test_users, local_product_counts_train, local_product_counts_dev, local_product_counts_test
train_users, dev_users, test_users, local_product_counts_train, local_product_counts_dev, local_product_counts_test = distribute_global_neighbors(
    users_list, num_users_train, num_users_dev, num_users_test, global_product_counts
)

def get_local_neighbor_count(users, local_product_counts):
    user_neighbor_counts = []

    for user_id, reviews in users:
        first_review = reviews[0]  # Focus on the first review in the profile
        asin = first_review['asin']
        local_neighbors = local_product_counts[asin] - 1  # Subtract 1 to exclude the current review
        user_neighbor_counts.append((user_id, local_neighbors))

    return pd.Series([count for _, count in user_neighbor_counts]).value_counts().sort_index()
def print_local_neighbor_distribution_with_percentages(local_neighbor_count, split_name):
    total_users = local_neighbor_count.sum()  # Total number of users

    # Print the distribution with percentages
    print(f"{split_name} Local Neighbor Count Distribution (for the first review):")
    for neighbor_count, count in local_neighbor_count.items():
        percentage = (count / total_users) * 100
        print(f"Users with {neighbor_count} neighbors: {count} ({percentage:.2f}%)")
    print("\n")

# Function to print the profile size distribution with percentages
def print_profile_size_distribution_with_percentages(profile_size_distribution, split_name):
    total_users = profile_size_distribution.sum()  # Total number of users

    # Print the distribution with percentages
    print(f"{split_name} Profile Size Distribution:")
    for profile_size, count in profile_size_distribution.items():
        percentage = (count / total_users) * 100
        print(f"Users with profile size {profile_size}: {count} ({percentage:.2f}%)")
    print("\n")

train_local_neighbor_count = get_local_neighbor_count(train_users, local_product_counts_train)
dev_local_neighbor_count = get_local_neighbor_count(dev_users, local_product_counts_dev)
test_local_neighbor_count = get_local_neighbor_count(test_users, local_product_counts_test)
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
        pred_entry = {"id": user_id, "output": reviews[0]['title']}

        # Add full user profile to user_entry
        user_entry["profile"] = [
            {
                'rating': d['rating'],
                'title': d['title'],
                'text': d['text'],
                # Uncomment the lines below if you want to include these fields
                # 'timestamp': d['timestamp'],
                # 'helpful_vote': d['helpful_vote'],
                # 'verified_purchase': d['verified_purchase'],
                 "productAsin": d['asin']
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
file_name_base = "amazon_title_generation"

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
