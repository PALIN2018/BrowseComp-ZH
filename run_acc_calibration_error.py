import argparse
import json
import os
import pandas as pd

MODELS = [
    "DeepSeek-V3", "DeepSeek-R1", "Qwen2.5-72B-Instruct", "QwQ-32B",
    "Llama4", "Qwen3-235B-thinking", "Qwen3-235B-no-thinking", "GPT-4o",
    "O1", "O4-mini", "Claude3.5-Sonnet", "Claude3.7-think", "Gemini2.0-Flash",
    "Gemini2.5-Pro", "Qwen2.5-Max"
]

CONFIDENCE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]

def process_model_data(model, input_path):
    """Process data for a single model"""
    file_path = os.path.join(input_path, model, 'infer.jsonl')
    with open(file_path) as f:
        records = [json.loads(line) for line in f]
    
    stats = {
        'correct': 0, 'wrong': 0, 'none': 0,
        'calibration': [{'samples':0, 'correct':0, 'conf_sum':0} for _ in CONFIDENCE_BINS],
        'confidences': [],
        'solutions': []
    }
    
    for record in records:
        eval_res = record['eval_result'][0]
        is_correct = eval_res['is_correct'].lower()
        confidence = int(eval_res['model_extracted_confidence'].split('%')[0])
        
        # Count correct/incorrect/None results
        if is_correct == 'yes':
            stats['correct'] += 1
        elif is_correct == 'no':
            stats['wrong'] += 1
        else:
            stats['none'] += 1
        
        # Update calibration statistics
        bin_idx = min(confidence // 20, len(CONFIDENCE_BINS)-1)
        bin_stats = stats['calibration'][bin_idx]
        bin_stats['samples'] += 1
        bin_stats['conf_sum'] += confidence
        if is_correct == 'yes':
            bin_stats['correct'] += 1
        
        # save origin data
        stats['confidences'].append(eval_res['model_extracted_confidence'])
        stats['solutions'].append(is_correct)
    
    return stats

def calculate_calibration(stats, total):
    """calculate calibration statistics"""
    error = 0.0
    for bin_stats in stats:
        samples = bin_stats['samples']
        if not samples:
            continue
        accuracy = bin_stats['correct'] / samples
        avg_conf = bin_stats['conf_sum'] / samples / 100  # convert to 0-1 decimal
        error += (samples / total) * abs(accuracy - avg_conf)
    return error * 100  # convert to percentage

def collect_infer_outcome(args):
    results = []
    
    for model in MODELS:
        data = process_model_data(model, args.input_path)
        total = data['correct'] + data['wrong'] + data['none']
        
        # build result
        model_result = {
            'Model': model,
            'Accuracy (%)': round(data['correct'] / total * 100, 4),
            'Calibration Error (%)': round(calculate_calibration(data['calibration'], total), 4),
            'Details': {
                'correct': data['correct'],
                'wrong': data['wrong'],
                'none': data['none'],
                'total': total,
                'solutions': data['solutions'],
                'confidences': data['confidences']
            }
        }
        results.append(model_result)
    
    # save results
    os.makedirs(args.output_path, exist_ok=True)
    
    # save JSON
    with open(os.path.join(args.output_path, 'infer_acc.json'), 'w') as f:
        json.dump({r['Model']: r['Details'] for r in results}, f, indent=4, ensure_ascii=False)
    
    # save CSV
    df = pd.DataFrame([{
        'Model': r['Model'],
        'Accuracy (%)': r['Accuracy (%)'],
        'Calibration Error (%)': r['Calibration Error (%)']
    } for r in results])
    df.to_csv(os.path.join(args.output_path, 'infer_acc.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model inference result statistics tool')
    parser.add_argument('--input_path', type=str, default="output_data")
    parser.add_argument('--output_path', type=str, default="outcome_data")
    args = parser.parse_args()
    
    collect_infer_outcome(args)