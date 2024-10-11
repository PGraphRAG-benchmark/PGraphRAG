from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import time
import torch
import json
from openai import AzureOpenAI
from google.colab import userdata
import os
import argparse
import sys

class UserProfile:
    def __init__(self, profile, dataset, split, task, ranker):

        self.dataset = dataset
        self.task = task
        self.ranker = ranker
        self.split = split

        self.user_id = profile['user_id']
        self.product_id = profile['product_id']
        self.user_review_text = profile['user_review_text']
        self.user_review_title = profile.get('user_review_title', None)

        self.user_ratings = []
        for review in profile['user_ratings']:
            self.user_ratings.append({"reviewTitle": review.get('reviewTitle', None), "reviewText": review.get("reviewText", None)})

        self.neighbor_ratings = []
        for review in profile['neighbor_ratings']:
            self.neighbor_ratings.append({"reviewTitle": review.get('reviewTitle', None), "reviewText": review.get("reviewText", None)})

        
        self.random_review = profile['random_review']


        '''
        # deprecated
        self.all_ratings = []
        for review in profile['all_ratings']:
            self.all_ratings.append({"reviewTitle": review.get('reviewTitle', None), "reviewText": review.get("reviewText", None)})
        '''



    # Retrieve relevant part of main review based on task, return as formatted string
    def get_review(self):

        if self.task == "reviewTitle":
            return f"Review text: '{self.user_review_text}'\n"

        elif self.task == "reviewText":
            return f"Review title: '{self.user_review_title}'\n"

        elif self.task == "reviewRating":
            return f"Review title: '{self.user_review_title}', Review text: '{self.user_review_text}'\n"


    # Retrieve related reviews from profile based on {mode} and {k}
    def retrieve(self, mode, k):

        if mode == "user":
            retrieved = "User's Own Reviews:\n"
            for review in self.user_ratings[:k]:
                rating = ""
                if self.task == "reviewRating":
                    rating += f", Review rating: {review['reviewRating']}"

                context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
                retrieved += context

            return retrieved

        elif mode == "neighbor":
            retrieved = "Other Users' Reviews:\n"
            for review in self.neighbor_ratings[:k]:
                rating = ""
                if self.task == "reviewRating":
                    rating += f", Review rating: {review['reviewRating']}"

                context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
                retrieved += context

            return retrieved

        elif mode == "random":
            retrieved = "Random Review:\n"
            review = self.random_review

            rating = ""
            if self.task == "reviewRating":
                rating += f", Review rating: {review['reviewRating']}"

            context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"{rating}\n"
            retrieved += context

            return retrieved            

        elif mode == "none":
            return ""

        

        '''
        # deprecated
        elif mode == "all":
            retrieved = "Other Users' Reviews:\n"
            for review in self.all_ratings[:k]:

                if self.dataset == "google":
                    context = f"Review text: \"{review['reviewText']}\"\n"
                else: # dataset == "amazon" or "b2w"
                    context = f"Review title: \"{review['reviewTitle']}\", Review text: \"{review['reviewText']}\"\n"
                retrieved += context
            return retrieved
        '''

    # Creates prompt for {task} on main review, with retrieval based on {mode} and {k}
    def create_prompt(self, mode, k):

        prompt = ""

        # Initialize intro based on mode
        if mode == "both":
            intro = "Given the following reviews from the same user and other users on the same product:\n"
        # elif mode == "all":
        #     intro = "Given the following reviews from any user on any product:\n"
        elif mode == "random":
            intro = "Given a random review from any user on any product:\n"
        elif mode == "user":
            intro = "Given the following reviews from the user on different products:\n"
        elif mode == "neighbor":
            intro = "Given the following reviews from other users on the same product:\n"
        elif mode == "none":
            intro = "Given only information on this review:\n"

        prompt += intro


        # Retrieve profiles based on mode
        if mode == "both":
            retrieved_profiles = f"{self.retrieve('user', k)}\n{self.retrieve('neighbor', k)}"

        else: # mode in ["user", "neighbor", "none", "all"]
            retrieved_profiles = self.retrieve(mode, k)

        prompt += retrieved_profiles


        # Set up directions based on task
        if self.task == "reviewTitle":
            direction = "\nGenerate a title for the following product review from this user without any explanation: "
            direction += self.get_review() # append reviewText for title generation
            direction += "Generate the review title in 10 words or less using the format: 'Review title:'."

        elif self.task == "reviewText": # ONLY FOR AMAZON AND B2W(, and yelp?)
            direction = "\nGenerate a review for the following product from this user given the review title, without any explanation: "
            direction += self.get_review() # append reviewTitle for text generation
            direction += "Generate the review text using the format: 'Review text:'."

        elif self.task == "reviewRating":
            direction = "\nGenerate an integer rating from 1-5 for the following product from this user given the review title and text, without any explanation: "
            direction += self.get_review() # append reviewTitle and reviewText for rating generation
            direction += "Generate the review rating using the format: 'Rating:'."

        prompt += direction

        return prompt


# Function to use GPT to generate given a {prompt}
def gpt_call(prompt, client):
    while True:

        try:
            response = client.chat.completions.create(
                model= "gpt-4o-mini-20240718",
                messages=[
                    {"role": "system", "content": "You are a personalized assistant, with the goal of providing users the best content using their preferences and the preferences of similar users."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4  # temp change????????????
            )

            # Extract and print the assistant's response from the first choice
            if response.choices:
                generated_text = response.choices[0].message.content
                #print(f"Generated text: {generated_text}") # if you want to see generations in output
                return generated_text

        except Exception as e:
            print(f"An error occurred in fetching the chat response: {e}")
            time.sleep(10)


# CAN be used to generate a SINGLE results file, specifying mode and k
# Function to generate {task} on {dataset}-{split} for 1 {mode} and 1 {k} with {ranker} using gpt
def generate_gpt(data, dataset, split, task, ranker, mode, k, client=None):
    print(f"Processing mode: {mode} with k={k} on GPT")

    if not client:
        client = AzureOpenAI(
            azure_endpoint = "https://vietgpt.openai.azure.com/",
            api_key=userdata.get('AZURE_KEY'),
            api_version="2024-02-15-preview"
            )

    results = []

    for profile in tqdm(data, desc=f'Generating for OUTPUT-{dataset}_{split}_{task}_GPT_{ranker}-{mode}_k{k}'):
        # Store user profile in a UserProfile object
        p = UserProfile(profile, dataset, split, task, ranker)

        # Synthesize prompt from profile based on task, mode, k
        prompt = p.create_prompt(mode, k)

        # Feed prompt to GPT and store response
        generation = gpt_call(prompt, client)
        print(generation) # IF you want to watch as generations run
        results.append(generation)

    # save results (PROBABLY WILL CHANGE)
    save_results(results, dataset, split, task, ranker, mode, k, "GPT")


    return


# CAN be used to generate a SINGLE results file, specifying mode and k
# Function to generate {task} on {dataset}-{split} for 1 {mode} and 1 {k} with {ranker} using llama
def generate_llama(data, dataset, split, task, ranker, mode, k, model=None):

    # (hard coded these in for now, not sure if you want it to be adaptable)
    max_input_length=512
    max_output_length=256

    print(f"Processing mode: {mode} with k={k} on LLAMA")

    if not model:
        model = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto",)
    #if not tokenizer:
    #    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    results = []

    for profile in tqdm(data, desc=f'Generating for OUTPUT-{dataset}_{split}_{task}_LLAMA_{ranker}-{mode}_k{k}'):
        # Store user profile in a UserProfile object
        p = UserProfile(profile, dataset, split, task, ranker)

        # Synthesize prompt from profile based on task, mode, k
        prompt = p.create_prompt(mode, k)

        llama_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            f"Do NOT generate anything else!.\n"
            f"<<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        # Feed prompt to LLAMA and store response
        generation = model(llama_prompt, max_new_tokens=max_output_length, do_sample=True, return_full_text=False)
        print(generation) # IF you want to watch as generations run
        results.append(generation)

    # save results (PROBABLY WILL CHANGE)
    save_results(results, dataset, split, task, ranker, mode, k, "LLAMA")

    return


# not necessary anymore since now partial_generate() is capable of doing full
'''
# Function to specify model and generate EVERYTHING for this {dataset} {task}
def full_generate(data, dataset, split, task, ranker, model):
    modes = ["none", "all", "user", "neighbor", "both"]
    k_values = [1, 2, 4]

    partial_generate(data, dataset, split, task, ranker, model, modes, k_values)
'''

# Function to generate on a subset of modes and/or a subset of k values
# Generates everything if modes+k_values are not specified
# def partial_generate(data, dataset, split, task, ranker, model, modes=["none", "all", "user", "neighbor", "both"], k_values=[1, 2, 4]):
def partial_generate(data, dataset, split, task, ranker, model, modes=["none", "user", "neighbor", "both", "random"], k_values=[1, 2, 4]):

    # use gpt to generate for all mode-k combinations
    if model == "gpt":
        gpt_client = AzureOpenAI(
            azure_endpoint = "https://vietgpt.openai.azure.com/",
            api_key=userdata.get('AZURE_KEY'),
            api_version="2024-02-15-preview"
            )

        for k in k_values:
            for mode in modes:
                generate_gpt(data, dataset, split, task, ranker, mode, k, client=gpt_client)


    # use llama to generate for all mode-k combinations
    elif model == "llama":
        llama3_model = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto",)
        #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

        for k in k_values:
            for mode in modes:
                generate_llama(data, dataset, split, task, ranker, mode, k, model=llama3_model)

# Function to load data from a (ranking) JSON file
# example: b2w_data_dev_ranked_k_5_reviewText_bm25.json

'''
#load data using old format
def load_data(file_path):
    # pull filename from path
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # parse run information from filename
    parsed = filename.split('_') # ['b2w', 'data', 'dev', 'ranked', 'k', '5', 'reviewText', 'bm25'] # remove k5?????????? if so, use below function

    dataset = parsed[0]
    task = parsed[6]
    ranker = parsed[7]
    split = parsed[2]
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data, dataset, split, task, ranker
'''


#load data using new format
def load_data(file_path):
    # pull filename from path
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Remove the 'RANKED-' prefix
    if filename.startswith("RANKING-"):
        filename = filename[len("RANKING-"):]

    # parse run information from filename
    parsed = filename.split('_') # ['b2w', 'dev', 'reviewText', 'bm25']

    dataset = parsed[0]
    split = parsed[1]
    task = parsed[2]
    ranker = parsed[3]
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data, dataset, split, task, ranker


def save_results(results, dataset, split, task, ranker, mode, k, model):
    directory = './results'
    filename = f'OUTPUT-{dataset}_{split}_{task}_{model}_{ranker}-{mode}_k{k}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"{model} results for {dataset}-{split}-{task} mode='{mode}' and k={k} on ranker='{ranker}' have been saved to {filepath}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generation Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to input data file")
    parser.add_argument('--model', type=str, choices=["gpt", "llama"], required=True, help="Model to use ('gpt' or 'llama')")
    parser.add_argument('--mode', nargs='+', type=str, choices=["none", "random", "user", "neighbor", "both"], help="Mode(s) to generate on. Leave empty if all modes")
    parser.add_argument('--k', nargs='+', type=int, help="K-value(s) to generate on. Leave empty if all k")

    args = parser.parse_args()

    args.model = args.model.lower()
    if args.model not in ['gpt', 'llama']:
        parser.error("Model must be 'gpt' or 'llama'")

    if not os.path.isfile(args.input):
        parser.error(f"Error: The file '{args.input}' does not exist.")

    return args

def main():

    # load args, corresponding model, data
    args = parse_arguments()
    data, dataset, split, task, ranker = load_data(args.input)


    if args.mode and args.k: # specify mode and k
        partial_generate(data, dataset, split, task, ranker, args.model, modes=args.mode, k_values=args.k)
    elif args.mode: # specify mode
        partial_generate(data, dataset, split, task, ranker, args.model, modes=args.mode)
    elif args.k: # specify k
        partial_generate(data, dataset, split, task, ranker, args.model, k_values=args.k)
    else: # run every mode, every k
        partial_generate(data, dataset, split, task, ranker, args.model)

if __name__ == "__main__":
    main()

