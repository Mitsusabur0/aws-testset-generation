import pandas as pd
import ast
import re
import os

# ==========================================
# 1. CONFIGURATION / HYPERPARAMETERS
# ==========================================
CONFIG = {
    "INPUT_FILENAME": "testsets/ragas_testset_simulated.csv",
    "OUTPUT_FILENAME": "testset_evaluations/evaluation_results_1.parquet",
    "TEXT_NORMALIZATION": True, # Set to False if you want strict exact case matching
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    """
    Normalizes text to ensure 'Context A' matches 'context a '.
    Removes newlines, extra spaces, and converts to lowercase.
    """
    if not isinstance(text, str):
        return ""
    
    if CONFIG["TEXT_NORMALIZATION"]:
        # Lowercase
        text = text.lower()
        # Remove newlines and tabs
        text = text.replace('\n', ' ').replace('\r', '')
        # Remove multiple spaces resulting from the previous step
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text

def parse_list_column(data):
    """
    Safely parses a stringified list (e.g., "['a', 'b']") into a Python list.
    Returns an empty list if parsing fails.
    """
    try:
        if isinstance(data, list):
            return data
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return []

def compute_metrics(row):
    """
    Calculates retrieval metrics for a single row.
    """
    # 1. Parse and Normalize Ground Truths
    # We use a set for GT to allow O(1) lookups, but keep list for duplicate checking if needed
    gt_raw = row.get('reference_contexts', [])
    gt_normalized = set([clean_text(txt) for txt in gt_raw])
    
    # 2. Parse and Normalize Retrieved Contexts
    # We keep the order of retrieved items for MRR calculation
    ret_raw = row.get('retrieved_contexts', [])
    ret_normalized = [clean_text(txt) for txt in ret_raw]
    
    # --- METRIC CALCULATIONS ---
    
    # A. HIT RATE (Binary: Did we find at least one correct answer?)
    # Check intersection
    hits = [ctx for ctx in ret_normalized if ctx in gt_normalized]
    hit_rate = 1 if len(hits) > 0 else 0
    
    # B. RECALL (What % of the Ground Truth did we find?)
    if len(gt_normalized) > 0:
        recall = len(hits) / len(gt_normalized)
    else:
        recall = 0.0

    # C. PRECISION (What % of retrieved items were actually relevant?)
    if len(ret_normalized) > 0:
        precision = len(hits) / len(ret_normalized)
    else:
        precision = 0.0

    # D. MEAN RECIPROCAL RANK (MRR)
    # Score is 1/rank of the FIRST relevant item. 
    # If first item is correct, score 1. If second is correct, score 0.5.
    mrr = 0.0
    for i, ctx in enumerate(ret_normalized):
        if ctx in gt_normalized:
            mrr = 1 / (i + 1)
            break # Stop after finding the first relevant item

    return pd.Series({
        'hit_rate': hit_rate,
        'mrr': mrr,
        'precision': precision,
        'recall': recall,
        'retrieved_count': len(ret_normalized),
        'gt_count': len(gt_normalized)
    })

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================

def main():
    print(f"Loading data from: {CONFIG['INPUT_FILENAME']}...")
    
    if not os.path.exists(CONFIG['INPUT_FILENAME']):
        print(f"Error: File {CONFIG['INPUT_FILENAME']} not found.")
        return

    # Load Data
    df = pd.read_csv(CONFIG['INPUT_FILENAME'])
    
    print(f"Processing {len(df)} rows...")

    # Step 1: Fix Data Types (String -> List)
    # Apply parsing explicitly to ensure we are working with lists
    df['reference_contexts'] = df['reference_contexts'].apply(parse_list_column)
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(parse_list_column)

    # Step 2: Calculate Metrics
    # apply(..., axis=1) sends each row to the function
    metrics_df = df.apply(compute_metrics, axis=1)
    
    # Step 3: Combine Original Data with Metrics
    final_df = pd.concat([df, metrics_df], axis=1)
    
    # Step 4: Summary Statistics
    print("\n--- Evaluation Summary ---")
    print(f"Average Hit Rate:  {final_df['hit_rate'].mean():.2%}")
    print(f"Average MRR:       {final_df['mrr'].mean():.4f}")
    print(f"Average Recall:    {final_df['recall'].mean():.2%}")
    print(f"Average Precision: {final_df['precision'].mean():.2%}")
    
    # Step 5: Save Output
    # We save as Parquet to preserve the List data types for the frontend
    final_df.to_parquet(CONFIG['OUTPUT_FILENAME'])
    print(f"\nResults saved to: {CONFIG['OUTPUT_FILENAME']}")
    print("You can now load this file into your Streamlit app.")

if __name__ == "__main__":
    main()