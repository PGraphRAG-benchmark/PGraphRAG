from tqdm import tqdm
from transformers import pipeline
import time
import json
import os
import argparse
import sys
from openai import AzureOpenAI

class UserProfile:
    def __init__(self, profile, dataset, split, task):
        self.dataset = dataset
        self.task = task
        self.split = split

        self.user_id = profile['user_id']
        self.product_id = profile['product_id']
        self.user_review_text = profile['user_review_text']
        self.user_review_title = profile.get('user_review_title', None)

        self.user_ratings = [
            {
                "reviewTitle": review.get("reviewTitle", None),
                "reviewText": review.get("reviewText", None),
                "reviewRating": review.get("reviewRating", None)
            }
            for review in profile['user_ratings']
        ]

        self.neighbor_ratings = [
            {
                "reviewTitle": review.get("reviewTitle", None),
                "reviewText": review.get("reviewText", None),
                "reviewRating": review.get("reviewRating", None)
            }
            for review in profile['neighbor_ratings']
        ]

        self.random_review = profile['random_review']

    # Retrieve relevant part of main review based on task, return as formatted string
    def get_review(self):
        if self.task == "reviewTitle":
            return f"Review text: '{self.user_review_text}'\n"
        elif self.task == "reviewText":
            return f"Review title: '{self.user_review_title}'\n"
        elif self.task == "reviewRating":
            return f"Review title: '{self.user_review_title}', Review text: '{self.user_review_text}'\n"

    # Retrieve related reviews from profile based on {mode}
    def retrieve(self, mode):
        """
        Returns all user or neighbor reviews (no [:k] slicing).
        """
        if mode == "user":
            retrieved = "User's Own Reviews:\n"
            for review in self.user_ratings:
                rating = ""
                if review and self.task == "reviewRating":
                    rating += f", Review rating: {review['reviewRating']}"
                context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
                retrieved += context
            return retrieved

        elif mode == "neighbor":
            retrieved = "Other Users' Reviews:\n"
            for review in self.neighbor_ratings:
                rating = ""
                if review and self.task == "reviewRating":
                    rating += f", Review rating: {review['reviewRating']}"
                context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
                retrieved += context
            return retrieved

        elif mode == "random":
            retrieved = "Random Review:\n"
            review = self.random_review
            rating = ""
            if review and self.task == "reviewRating":
                rating += f", Review rating: {review['reviewRating']}"
            context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
            retrieved += context
            return retrieved

        elif mode == "none":
            return ""

    # Creates prompt for {task} on main review, with retrieval based on {mode}
    def create_prompt(self, mode):
        prompt = ""

        # Intro message
        if mode == "both":
            intro = "Given the following reviews from the same user and other users on the same product:\n"
        elif mode == "random":
            intro = "Given a random review from any user on any product:\n"
        elif mode == "user":
            intro = "Given the following reviews from the user on different products:\n"
        elif mode == "neighbor":
            intro = "Given the following reviews from other users on the same product:\n"
        elif mode == "none":
            intro = "Given only information on this review:\n"

        prompt += intro

        # Retrieve profiles
        if mode == "both":
            # combine user & neighbor
            retrieved_profiles = f"{self.retrieve('user')}\n{self.retrieve('neighbor')}"
        else:  # mode in ["user", "neighbor", "none", "random"]
            retrieved_profiles = self.retrieve(mode)

        prompt += retrieved_profiles

        # If the dataset is Portuguese, add a note
        portuguese = ""
        if self.dataset == "b2w":
            portuguese = "in Portuguese "

        # Directions based on task
        if self.task == "reviewTitle":
            direction = (
                f"\nGenerate a title {portuguese}for the following product review "
                f"from this user without any explanation: "
            )
            direction += self.get_review()
            direction += "Generate the review title in 10 words or less using the format: 'Review title:'."

        elif self.task == "reviewText":
            direction = (
                f"\nGenerate a review {portuguese}for the following product from this user "
                f"given the review title, without any explanation: "
            )
            direction += self.get_review()
            direction += "Generate the review text using the format: 'Review text:'."

        elif self.task == "reviewRating":
            direction = (
                "\nGenerate an integer rating from 1-5 for the following product from this user "
                "given the review title and text, without any explanation: "
            )
            direction += self.get_review()
            direction += "Generate the integer review rating using the format: 'Rating:'."

        prompt += direction
        return prompt


def gpt_call(prompt, client):
    while True:
        try:
            # Example with hypothetical client
            response = client.chat.completions.create(
                model="gpt-4o-mini-20240718",  # Replace with your model
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a personalized assistant, with the goal of providing users "
                            "the best content using their preferences and the preferences of similar users."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            if response.choices:
                generated_text = response.choices[0].message.content
                return generated_text

        except Exception as e:
            print(f"An error occurred in fetching the chat response: {e}")
            time.sleep(10)


def generate_gpt(data, dataset, split, task, mode, client=None):
    """
    Generates results using GPT for the specified mode (no ranker).
    """
    print(f"Processing mode: {mode} on GPT")

    if not client:
        client = AzureOpenAI(
            azure_endpoint = "https://vietgpt.openai.azure.com/", #Replace with your endpoint
            api_key = os.environ.get("AZURE_KEY"),
            api_version="2024-02-15-preview"
            )

    results = []
    for profile in tqdm(data, desc=f'Generating GPT output for {dataset}_{split}_{task}-{mode}'):
        p = UserProfile(profile, dataset, split, task)
        prompt = p.create_prompt(mode)
        generation = gpt_call(prompt, client)
        print(generation)
        results.append({'user_id': p.user_id, 'output': generation})

    save_results(results, dataset, split, task, mode, model="GPT")


def generate_llama(data, dataset, split, task, mode, model=None):
    """
    Generates results using a LLaMA pipeline for the specified mode (no ranker).
    """
    max_output_length = 256
    print(f"Processing mode: {mode} on LLAMA")

    if not model:
        model = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto")

    results = []
    for profile in tqdm(data, desc=f'Generating LLAMA output for {dataset}_{split}_{task}-{mode}'):
        p = UserProfile(profile, dataset, split, task)
        prompt = p.create_prompt(mode)

        llama_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            f"Do NOT generate anything else!.\n"
            f"<<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        generation = model(llama_prompt, max_new_tokens=max_output_length, do_sample=True, return_full_text=False)
        text_output = generation[0]['generated_text']
        print(text_output)
        results.append({'user_id': p.user_id, 'output': text_output})

    save_results(results, dataset, split, task, mode, model="LLAMA")


def load_data(file_path):
    """
    Minimal JSON loader without ranker logic.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_results(results, dataset, split, task, mode, model):
    directory = f'./results/{dataset}/{split}/{task}/{model}'
    filename = f'OUTPUT-{dataset}_{split}_{task}_{model}-{mode}.json'

    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"{model} results for {dataset}-{split}-{task}, mode='{mode}' have been saved to {filepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generation Pipeline")

    # Path to your JSON file
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Path to input data file (JSON)"
    )
    
    # This helps with folder naming if you want to keep them
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='amazon', 
        help="Name of dataset (e.g. amazon, b2w, etc.)"
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        default='test', 
        help="Which split? (e.g. train, dev, test)"
    )
    
    # The key argument: which task to run
    parser.add_argument(
        '--task', 
        type=str, 
        required=True, 
        choices=["reviewTitle", "reviewText", "reviewRating"],
        help="Which task to run (e.g. reviewTitle, reviewText, reviewRating)"
    )
    
    # Mode(s) to generate on. 
    # Accepts multiple modes (nargs='+'), or none means we do them all.
    parser.add_argument(
        '--mode', 
        nargs='+', 
        type=str, 
        choices=["none", "random", "user", "neighbor", "both"],
        help="Which retrieval mode(s) to run. Leave empty to run them all."
    )

    # Which model to use, gpt or llama
    parser.add_argument(
        '--model',
        type=str,
        choices=["gpt", "llama"],
        required=True,
        help="Model to use ('gpt' or 'llama')"
    )

    args = parser.parse_args()

    # Validate the input file
    if not os.path.isfile(args.input):
        parser.error(f"Error: The file '{args.input}' does not exist.")

    return args

def main():
    # Parse the command-line arguments
    args = parse_arguments()

    # Load data from the specified file
    data = load_data(args.input)

    # If user didn't specify any mode(s), we run all
    if args.mode:
        modes_to_run = args.mode
    else:
        modes_to_run = ["none", "random", "user", "neighbor", "both"]

    # Based on which model was requested, call the relevant function
    if args.model == "gpt":
        for mode in modes_to_run:
            generate_gpt(data, args.dataset, args.split, args.task, mode)
    else:  # 'llama'
        # Optionally load a LLaMA pipeline once
        llama_pipeline = pipeline(
            "text-generation", 
            model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
            device_map="auto"
        )
        for mode in modes_to_run:
            generate_llama(data, args.dataset, args.split, args.task, mode, model=llama_pipeline)

if __name__ == "__main__":
    main()
