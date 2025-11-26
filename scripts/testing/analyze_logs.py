import csv
import sys
import os
from collections import Counter, defaultdict

# Determine paths dynamically
# Script is in scripts/testing/, so project root is ../../
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

LOG_FILE = os.path.join(OUTPUT_DIR, "crema_full_validation_log.csv")
ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "crema_full_analysis.txt")

def log_print(f, text):
    print(text)
    f.write(text + "\n")

if not os.path.exists(LOG_FILE):
    print(f"Error: {LOG_FILE} not found.")
    print("Make sure you run this script from the project root or scripts/testing folder.")
    sys.exit(1)

try:
    total = 0
    correct = 0
    by_emotion = defaultdict(lambda: {"total": 0, "correct": 0})
    confusions = Counter()

    errors = []

    with open(LOG_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            true_emo = row['True Emotion']
            pred_emo = row['Predicted Emotion']
            filename = row['File']
            
            by_emotion[true_emo]["total"] += 1
            
            if row['Result'] == 'CORRECT':
                correct += 1
                by_emotion[true_emo]["correct"] += 1
            else:
                confusions[(true_emo, pred_emo)] += 1
                errors.append((filename, true_emo, pred_emo))

    if total == 0:
        print("No data found in log file.")
        sys.exit(0)

    # Open output file
    with open(ANALYSIS_FILE, "w") as out_f:
        # Overall Accuracy
        accuracy = (correct / total) * 100
        
        log_print(out_f, f"--- CREMA-D Full Validation Analysis ---")
        log_print(out_f, f"Log File: {LOG_FILE}")
        log_print(out_f, f"Total Samples: {total}")
        log_print(out_f, f"Overall Accuracy: {accuracy:.2f}%")
        
        # Per-Class Accuracy
        log_print(out_f, "\n--- Per-Class Accuracy ---")
        for emo in sorted(by_emotion.keys()):
            stats = by_emotion[emo]
            sub_acc = (stats["correct"] / stats["total"]) * 100
            log_print(out_f, f"{emo.ljust(10)}: {sub_acc:.2f}% ({stats['correct']}/{stats['total']})")
            
        # Confusion Matrix (Simplified)
        log_print(out_f, "\n--- Common Confusions (True -> Predicted) ---")
        for (true_e, pred_e), count in confusions.most_common(10):
            log_print(out_f, f"{true_e} -> {pred_e}: {count} times")

        # Detailed Errors
        log_print(out_f, "\n--- Misclassified Files (Details) ---")
        log_print(out_f, f"Total Errors: {len(errors)}")
        log_print(out_f, f"{'Filename'.ljust(30)} | {'True'.ljust(10)} -> {'Predicted'.ljust(10)}")
        log_print(out_f, "-" * 60)
        
        for fname, true_e, pred_e in errors:
            log_print(out_f, f"{fname.ljust(30)} | {true_e.ljust(10)} -> {pred_e.ljust(10)}")


    print(f"\nAnalysis saved to: {ANALYSIS_FILE}")

except Exception as e:
    print(f"Analysis failed: {e}")
