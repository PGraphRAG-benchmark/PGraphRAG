import argparse
import json
import re
import os
import csv

from evaluate import load
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--gold', type=str, required=True,
                        help="Path to gold labels JSON file")
    parser.add_argument('--preds', type=str, required=True,
                        help="Path to predictions JSON file or folder of JSON files")
    parser.add_argument('--task', type=str, required=True,
                        choices=["reviewTitle", "reviewText", "reviewRating"],
                        help="Which task to evaluate (e.g. reviewTitle, reviewText, reviewRating)")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="Directory to save the evaluation CSV (default: current dir)")

    return parser.parse_args()

def load_data(gold_path, pred_path):
    """
    Loads gold labels and predictions from JSON.
    gold_path: JSON file
    pred_path: JSON file
    returns: (list_of_golds, list_of_preds)
    """
    with open(gold_path, 'r') as f:
        gold_labels = json.load(f)

    with open(pred_path, 'r') as f:
        predictions = json.load(f)

    return gold_labels, predictions

def extract_prediction(pred_text, task):
    """
    Extract the final string from the model's output depending on the task.
    Adjust logic as needed based on your prompt format.
    """
    noid = re.sub(r'\[\[VIDEOID:[^\]]*\]\]', '', pred_text)  # example cleaning

    if task == 'reviewText':
        # Attempt different known patterns
        if '**Review text:** "' in noid:
            return noid.split('**Review text:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review text: "' in noid:
            return noid.split('Review text: "')[1].rsplit('"', 1)[0].strip()
        elif 'Review text: ' in noid:
            return noid.split('Review text: ')[1].strip()
        elif 'The review text is: "' in noid:
            return noid.split('The review text is: "')[1].rsplit('"', 1)[0].strip()
        elif 'The review text is: ' in noid:
            return noid.split('The review text is: ')[1].strip()
        else:
            return noid.strip()

    elif task == 'reviewTitle':
        if '**Review title:** "' in noid:
            return noid.split('**Review title:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review title: ' in noid:
            return noid.split('Review title: ')[1].strip()
        else:
            return noid.strip()

    elif task == 'reviewRating':
        if '**Review rating:** "' in noid:
            return noid.split('**Review rating:** "')[1].rsplit('"', 1)[0].strip()
        elif 'Review rating: ' in noid:
            return noid.split('Review rating: ')[1].strip()
        elif 'Rating: ' in noid:
            return noid.split('Rating: ')[1].strip()
        else:
            return noid.strip()

    # Default fallback
    return noid.strip()


def clean_data(golds, preds, task):
    """
    Given the gold labels and predictions for a single task,
    return two lists of "clean" strings for evaluation.
    """
    clean_golds = []
    clean_preds = []

    # Decide which gold field to compare based on the task:
    if task == 'reviewText':
        gold_field = 'user_review_text'
    elif task == 'reviewTitle':
        gold_field = 'user_review_title'
    elif task == 'reviewRating':
        gold_field = 'user_review_rating'
    else:
        raise ValueError(f"Unknown task: {task}")

    for i, pred in enumerate(preds):
        clean_golds.append(golds[i][gold_field].strip())
        clean_preds.append(extract_prediction(pred["output"], task))

    return clean_golds, clean_preds

def evaluate_predictions(golds, preds, rouge, meteor):
    """
    Run ROUGE and METEOR on the given gold and predicted lists.
    Returns a dict of metric results (rouge1, rougeL, meteor).
    """
    # Evaluate with ROUGE
    rouge_results = rouge.compute(predictions=preds, references=golds)
    # Evaluate with METEOR
    meteor_results = meteor.compute(predictions=preds, references=golds)

    return {
        'rouge1': round(rouge_results["rouge1"], 3),
        'rougeL': round(rouge_results["rougeL"], 3),
        'meteor': round(meteor_results["meteor"], 3)
    }

def main():
    args = parse_arguments()

    # Load metric objects
    rouge = load('rouge')
    meteor = load('meteor')

    # Check if preds is a file or directory
    all_results = []  # store multiple rows if directory
    if os.path.isfile(args.preds):
        # Evaluate a single predictions file
        gold_labels, predictions = load_data(args.gold, args.preds)
        clean_golds, clean_preds = clean_data(gold_labels, predictions, args.task)
        metrics = evaluate_predictions(clean_golds, clean_preds, rouge, meteor)

        # We can store the row with the preds filename + metrics
        row = [os.path.basename(args.preds), metrics['rouge1'], metrics['rougeL'], metrics['meteor']]
        all_results.append(row)

    elif os.path.isdir(args.preds):
        # Evaluate all JSON files inside the directory
        for filename in os.listdir(args.preds):
            if filename.lower().endswith('.json'):
                pred_file = os.path.join(args.preds, filename)
                gold_labels, predictions = load_data(args.gold, pred_file)
                clean_golds, clean_preds = clean_data(gold_labels, predictions, args.task)
                metrics = evaluate_predictions(clean_golds, clean_preds, rouge, meteor)

                # We can store the row with the preds filename + metrics
                row = [filename, metrics['rouge1'], metrics['rougeL'], metrics['meteor']]
                all_results.append(row)

    else:
        print(f"Error: '{args.preds}' is neither a file nor a directory.")
        return

    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename/Mode", "ROUGE-1", "ROUGE-L", "METEOR"])  # header
        writer.writerows(all_results)

    print(f"Evaluation results saved to {out_csv_path}")


if __name__ == "__main__":
    main()
