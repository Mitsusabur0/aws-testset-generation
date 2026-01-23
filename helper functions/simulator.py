import pandas as pd
import ast
import random
import io

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

# I will assume the CSV text you pasted is saved as 'ragas_testset_50.csv'
# If you don't have the file yet, simply paste your CSV content into a file named 'ragas_testset_50.csv'
INPUT_FILE = "testsets/ragas_testset_50.csv"
OUTPUT_FILE = "testsets/ragas_testset_simulated.csv"

# Set a seed so the "randomness" is the same every time we run this (for consistency)
random.seed(42)

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# ==========================================
# 2. CREATE THE "DISTRACTOR BANK"
# ==========================================
# In a real RAG system, "wrong" answers come from other documents in your database.
# We will simulate this by collecting all contexts from the file into one big pool.

all_contexts_pool = []

def collect_contexts(x):
    try:
        # Convert string "['abc']" to list ['abc']
        return ast.literal_eval(x)
    except:
        return []

# Parse the reference column and build the pool
parsed_references = df['reference_contexts'].apply(collect_contexts)
for ctx_list in parsed_references:
    for ctx in ctx_list:
        if isinstance(ctx, str) and len(ctx) > 10: # Only keep valid text
            all_contexts_pool.append(ctx)

# Remove duplicates to keep the pool clean
distractor_pool = list(set(all_contexts_pool))
print(f"Created a pool of {len(distractor_pool)} unique context snippets.")

# ==========================================
# 3. SIMULATION LOGIC
# ==========================================

def simulate_retrieval(row):
    # Get the True Ground Truth (GT)
    gt_list = collect_contexts(row['reference_contexts'])
    
    # If the GT is empty for some reason, just return 3 random distractors
    if not gt_list:
        return random.sample(distractor_pool, 3)
    
    true_context = gt_list[0] # We take the first GT as the "Gold" answer
    
    # Pick 3 random "wrong" contexts from the pool
    # We ensure we don't accidentally pick the true_context as a distractor
    noise = random.sample([x for x in distractor_pool if x != true_context], 3)
    
    # --- PROBABILITY DISTRIBUTION ---
    roll = random.random() # Number between 0.0 and 1.0
    
    if roll < 0.60:
        # SCENARIO A: GOOD RETRIEVAL (60%)
        # The right answer is at the top (Rank 1)
        # Result: [Correct, Noise, Noise]
        retrieved = [true_context, noise[0], noise[1]]
        
    elif roll < 0.80:
        # SCENARIO B: WEAK RETRIEVAL (20%)
        # The right answer is found, but buried (Rank 2 or 3)
        # Result: [Noise, Correct, Noise] OR [Noise, Noise, Correct]
        retrieved = [noise[0], noise[1], noise[2]]
        insert_pos = random.choice([1, 2]) # Index 1 or 2
        retrieved[insert_pos] = true_context
        
    else:
        # SCENARIO C: FAILURE / HALLUCINATION (20%)
        # The right answer is NOT found.
        # Result: [Noise, Noise, Noise]
        retrieved = [noise[0], noise[1], noise[2]]
        
    return retrieved

# ==========================================
# 4. APPLY AND SAVE
# ==========================================

print("Simulating retrieval results (K=3)...")
df['retrieved_contexts'] = df.apply(simulate_retrieval, axis=1)

print(f"Saving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print("Done! You can now run 'evaluate_rag.py' using this new file.")