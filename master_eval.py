from collections import Counter
from evaluate import load
import tqdm
import json
import re
import os
import csv

def load_data(golds, preds):
    with open(golds) as f:
        gold_labels = json.load(f)

    with open(preds) as f:
        predictions = json.load(f)

    return gold_labels, predictions


def extract_prediction(pred, task):
    if task == 'reviewText':
        pattern = r'\[\[VIDEOID:[^\]]*\]\]'
        noid = re.sub(pattern, '', pred)
        
        if '**Review text:** "' in noid:
            return noid.split('**Review text:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review text: "' in noid:
            return noid.split('Review text: "')[1].rsplit('"', 1)[0].strip()
        elif 'Review text: ' in noid:
            return noid.split('Review text: ')[1].strip()
        elif 'The review text is: \"' in noid:
            return noid.split('The review text is: \"')[1].rsplit('\"', 1)[0].strip()
        elif 'The review text is: "' in noid:
            return noid.split('The review text is: "')[1].rsplit('"', 1)[0].strip()
        elif 'The review text is: ' in noid:
            return noid.split('The review text is: ')[1].strip()
        else:
            return noid.strip()

    elif task == 'reviewTitle':
        if '**Review title:** ""' in pred:
            return pred.split('**Review title:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review title: ' in pred:
            return pred.split('Review title: ')[1].strip()
        else:
            return pred.strip()

    elif task == 'reviewRating':
        if '**Review rating:** ""' in pred:
            return pred.split('**Review rating:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review rating: ' in pred:
            return pred.split('Review rating: ')[1].strip()
        elif 'Rating: ' in pred:        
            return pred.split('Rating: ')[1].strip()
        else:
            return pred.strip()


# def clean_data(golds, preds, neighbors=None, model=None, dic=None):
def clean_data(golds, preds, task):
    clean_golds = []
    clean_preds = []


    if task == 'reviewText':
        field = 'user_review_text'
    elif task == 'reviewTitle':
        field = 'user_review_title'
    elif task == 'reviewRating':
        field = 'user_review_rating'


    for i, pred in enumerate(preds):
        clean_golds.append(golds[i][field].strip())
        clean_preds.append(extract_prediction(pred["output"], task))

    return clean_golds, clean_preds

def full_eval(golds, preds, mode_k, rouge, meteor):

    rouge_results = rouge.compute(predictions=preds, references=golds)
    meteor_results = meteor.compute(predictions=preds, references=golds)


    results = [
        mode_k, 
        round(rouge_results["rouge1"], 3), 
        round(rouge_results["rougeL"], 3), 
        round(meteor_results["meteor"], 3)
        ]

    return results

def dump_results(results, dataset, split, task, model, ranker, directory):
    header_line = f'Evaluation report for {dataset}_{split}_{task}_{model}_{ranker}'
    data_headers = ['mode_k', 'ROUGE-1', 'ROUGE-L', 'METEOR']

    filename = f'EVAL-{dataset}_{split}_{task}_{model}_{ranker}.csv'
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)

        writer.writerow([header_line])
        writer.writerow([])
        writer.writerow(data_headers)
        writer.writerows(results)

    print(f"Evaluation results saved to {file_path}")

    import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--ranking', type=str, required=True, help="Path to input data file")
    parser.add_argument('--results', type=str, required=True, help="Path to results file")

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    ranking_file = os.path.splitext(os.path.basename(args.ranking))[0]

    # Remove the 'RANKED-' prefix
    if ranking_file.startswith("RANKING-"):
        ranking_file = ranking_file[len("RANKING-"):]
    # parse run information from filename
    parsed = ranking_file.split('_')
    
    dataset = parsed[0]
    split = parsed[1]
    task = parsed[2]
    ranker = parsed[3]
    
    model = args.results.split('_')[3] # pulls model used for output

    rouge = load('rouge')
    meteor = load('meteor')

    if os.path.isdir(args.results):
        directory = args.results
        # Handle the case where input_path is a directory
        all_metrics = []
        for filename in os.listdir(args.results):
            file_path = os.path.join(args.results, filename)
            if os.path.isfile(file_path):
                currfile = os.path.splitext(os.path.basename(file_path))[0]
                mode_k = currfile.split('-')[-1] # pulls the mode and k from filename

                golds, preds = load_data(args.ranking, file_path)
                clean_golds, clean_preds = clean_data(golds, preds, task)
                all_metrics.append(full_eval(clean_golds, clean_preds, mode_k, rouge, meteor))
 
    elif os.path.isfile(args.results):
        directory = os.path.dirname(file_path)
        # Handle the case where input_path is a single file
        filename = os.path.splitext(os.path.basename(args.ranking))[0]
        mode_k = filename.split('-')[-1]

        golds, preds = load_data(args.ranking, filename)
        clean_golds, clean_preds = clean_data(golds, preds, task)
        all_metrics = [full_eval(clean_golds, clean_preds, mode_k, rouge, meteor)]
        
    else:
        print(f"Error: The provided path '{args.results}' is neither a file nor a directory.")


    dump_results(all_metrics, dataset, split, task, model, ranker, directory)

if __name__ == "__main__":
    main()