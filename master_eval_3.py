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
    Returns a dict of metric results (rouge1, rougeL, meteor) and per-review scores.
    """
    # Compute ROUGE with per-instance scores
    rouge_results = rouge.compute(predictions=preds, references=golds, use_aggregator=False)

    # Compute METEOR individually for each review
    meteor_scores = []
    for pred, gold in zip(preds, golds):
        meteor_score = meteor.compute(predictions=[pred], references=[gold])["meteor"]
        meteor_scores.append(meteor_score)

    # Extract per-review scores
    per_review_scores = []
    for i in range(len(preds)):
        rouge1_score = rouge_results["rouge1"][i]  # Now a list of per-instance scores
        rougeL_score = rouge_results["rougeL"][i]
        meteor_score = meteor_scores[i]  # Use manually computed per-instance METEOR
        avg_score = (rouge1_score + rougeL_score + meteor_score) / 3  # Compute average score

        per_review_scores.append({
            "index": i,
            "gold": golds[i],
            "pred": preds[i],
            "rouge1": round(rouge1_score, 3),
            "rougeL": round(rougeL_score, 3),
            "meteor": round(meteor_score, 3),
            "avg_score": round(avg_score, 3)
        })

    # Compute overall metrics (aggregate scores manually)
    overall_scores = {
        'rouge1': round(sum(rouge_results["rouge1"]) / len(preds), 3),
        'rougeL': round(sum(rouge_results["rougeL"]) / len(preds), 3),
        'meteor': round(sum(meteor_scores) / len(preds), 3)  # Manually computed average METEOR
    }

    return overall_scores, per_review_scores



def main():
    args = parse_arguments()

    # Load metric objects
    rouge = load('rouge')
    meteor = load('meteor')

    # Check if preds is a file or directory
    all_results = []  # store multiple rows if directory
    if os.path.isfile(args.preds):
        gold_labels, predictions = load_data(args.gold, args.preds)
        clean_golds, clean_preds = clean_data(gold_labels, predictions, args.task)
        overall_metrics, per_review_scores = evaluate_predictions(clean_golds, clean_preds, rouge, meteor)

        # Sort reviews by highest average score
        top_reviews = sorted(per_review_scores, key=lambda x: x["avg_score"], reverse=True)[:5]

        # Print Top 5 reviews
        print("\nTop 5 Reviews with Highest Scores:")
        for review in top_reviews:
            print(f"\nIndex: {review['index']}")
            print(f"Gold: {review['gold']}")
            print(f"Pred: {review['pred']}")
            print(f"ROUGE-1: {review['rouge1']}, ROUGE-L: {review['rougeL']}, METEOR: {review['meteor']}, Avg: {review['avg_score']}")

        # Store overall metrics
        row = [os.path.basename(args.preds), overall_metrics['rouge1'], overall_metrics['rougeL'], overall_metrics['meteor']]
        all_results.append(row)

    elif os.path.isdir(args.preds):
        for filename in os.listdir(args.preds):
            if filename.lower().endswith('.json'):
                pred_file = os.path.join(args.preds, filename)
                gold_labels, predictions = load_data(args.gold, pred_file)
                clean_golds, clean_preds = clean_data(gold_labels, predictions, args.task)
                overall_metrics, per_review_scores = evaluate_predictions(clean_golds, clean_preds, rouge, meteor)

                # Sort reviews by highest average score
                top_reviews = sorted(per_review_scores, key=lambda x: x["avg_score"], reverse=True)[:5]

                print(f"\nTop 5 Reviews for {filename}:")
                for review in top_reviews:
                    print(f"\nIndex: {review['index']}")
                    print(f"Gold: {review['gold']}")
                    print(f"Pred: {review['pred']}")
                    print(f"ROUGE-1: {review['rouge1']}, ROUGE-L: {review['rougeL']}, METEOR: {review['meteor']}, Avg: {review['avg_score']}")

                # Store overall metrics
                row = [filename, overall_metrics['rouge1'], overall_metrics['rougeL'], overall_metrics['meteor']]
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
