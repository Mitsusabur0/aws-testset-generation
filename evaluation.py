import pandas as pd
import ast
import re
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "INPUT_FILENAME": "testset_with_clean_retrieval.csv",
    "OUTPUT_FILENAME": "evaluations/testset_results.parquet",
    # Normalization helps ignore extra spaces or newlines when comparing substring
    "TEXT_NORMALIZATION": True, 
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    """
    Normalizes text to ensure robust substring matching.
    Lowercases and removes newlines/extra spaces.
    """
    if not isinstance(text, str):
        return ""
    
    if CONFIG["TEXT_NORMALIZATION"]:
        text = text.lower()
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', '')
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text

def parse_list_column(data):
    """
    Safely parses a stringified list (e.g., "['a', 'b']") into a Python list.
    """
    try:
        if isinstance(data, list):
            return data
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return []

def is_match(retrieved_item, ground_truths):
    """
    CRITICAL LOGIC CHANGE:
    Returns True if the 'retrieved_item' is found INSIDE any of the 'ground_truths'.
    This handles the case where Reference has a Footer/Link but Retrieval does not.
    """
    for gt in ground_truths:
        # Check if the clean retrieved text is a substring of the clean ground truth
        if retrieved_item in gt: 
            return True
    return False

def compute_metrics(row):
    """
    Calculates retrieval metrics for a single row using Substring Matching.
    """
    # 1. Parse and Normalize Ground Truths
    gt_raw = row.get('reference_contexts', [])
    gt_normalized = [clean_text(txt) for txt in gt_raw]
    
    # 2. Parse and Normalize Retrieved Contexts
    ret_raw = row.get('retrieved_contexts', [])
    ret_normalized = [clean_text(txt) for txt in ret_raw]
    
    # If no retrieval or no GT, return zeros (avoid division by zero)
    if not ret_normalized or not gt_normalized:
        return pd.Series({
            'hit_rate': 0, 'mrr': 0.0, 'precision': 0.0, 'recall': 0.0,
            'retrieved_count': len(ret_normalized), 'gt_count': len(gt_normalized)
        })

    # --- METRIC CALCULATIONS ---
    
    # Identify which retrieved items are "relevant" (matches)
    # Result is a list of Booleans: [True, False, True]
    matches_mask = [is_match(ctx, gt_normalized) for ctx in ret_normalized]
    
    total_matches = sum(matches_mask)

    # A. HIT RATE (Did we find at least one correct answer?)
    hit_rate = 1 if total_matches > 0 else 0
    
    # B. PRECISION (% of retrieved items that are correct)
    precision = total_matches / len(ret_normalized)

    # C. RECALL (How many of the GTs did we find?)
    # Note: Logic assumes if we matched a GT, we found it. 
    # Since we use substring, we check how many unique GTs were covered by our retrievals.
    covered_gts = 0
    for gt in gt_normalized:
        # Check if this specific GT was covered by ANY retrieval
        found = False
        for ret in ret_normalized:
            if ret in gt:
                found = True
                break
        if found:
            covered_gts += 1
            
    recall = covered_gts / len(gt_normalized)

    # D. MEAN RECIPROCAL RANK (MRR)
    # Score based on the rank of the *first* correct match.
    mrr = 0.0
    for i, is_correct in enumerate(matches_mask):
        if is_correct:
            mrr = 1 / (i + 1)
            break 

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
    # Ensure output directory exists
    os.makedirs(os.path.dirname(CONFIG['OUTPUT_FILENAME']), exist_ok=True)

    print(f"Loading data from: {CONFIG['INPUT_FILENAME']}...")
    
    if not os.path.exists(CONFIG['INPUT_FILENAME']):
        print(f"❌ Error: File {CONFIG['INPUT_FILENAME']} not found.")
        return

    # Load Data
    df = pd.read_csv(CONFIG['INPUT_FILENAME'])
    print(f"Processing {len(df)} rows...")

    # Step 1: Ensure columns are Lists (not strings)
    df['reference_contexts'] = df['reference_contexts'].apply(parse_list_column)
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(parse_list_column)

    # Step 2: Calculate Metrics
    metrics_df = df.apply(compute_metrics, axis=1)
    
    # Step 3: Combine Original Data with Metrics
    final_df = pd.concat([df, metrics_df], axis=1)
    
    # Step 4: Summary Statistics
    print("\n--- Evaluation Summary ---")
    print(f"Average Hit Rate:  {final_df['hit_rate'].mean():.2%}")
    print(f"Average MRR:       {final_df['mrr'].mean():.4f}")
    print(f"Average Precision: {final_df['precision'].mean():.2%}")
    print(f"Average Recall:    {final_df['recall'].mean():.2%}")
    
    # Step 5: Save Output
    final_df.to_parquet(CONFIG['OUTPUT_FILENAME'])
    print(f"\n✅ Results saved to: {CONFIG['OUTPUT_FILENAME']}")
    print("Ready for Streamlit.")

if __name__ == "__main__":
    main()