import csv
import sys
import os
from collections import Counter, defaultdict

log_file = "outputs/crema_full_validation_log.csv"

if not os.path.exists(log_file):
    print(f"Error: {log_file} not found.")
    sys.exit(1)

try:
    total = 0
    correct = 0
    by_emotion = defaultdict(lambda: {"total": 0, "correct": 0})
    confusions = Counter()

    with open(log_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            true_emo = row['True Emotion']
            pred_emo = row['Predicted Emotion']
            
            by_emotion[true_emo]["total"] += 1
            
            if row['Result'] == 'CORRECT':
                correct += 1
                by_emotion[true_emo]["correct"] += 1
            else:
                confusions[(true_emo, pred_emo)] += 1

    if total == 0:
        print("No data found in log file.")
        sys.exit(0)

    # Overall Accuracy
    accuracy = (correct / total) * 100
    
    print(f"--- Summary ---")
    print(f"Total Samples: {total}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Per-Class Accuracy
    print("\n--- Per-Class Accuracy ---")
    for emo in sorted(by_emotion.keys()):
        stats = by_emotion[emo]
        sub_acc = (stats["correct"] / stats["total"]) * 100
        print(f"{emo.ljust(10)}: {sub_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
    # Confusion Matrix (Simplified)
    print("\n--- Common Confusions (True -> Predicted) ---")
    for (true_e, pred_e), count in confusions.most_common(10):
        print(f"{true_e} -> {pred_e}: {count} times")

except Exception as e:
    print(f"Analysis failed: {e}")
