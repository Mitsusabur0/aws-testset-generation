import boto3
import pandas as pd
import re
from tqdm import tqdm

# --- CONFIGURATION ---
PROFILE = 'sandbox'
REGION = 'us-east-1'
KB_ID = '3TPM53DPBN' 

INPUT_CSV = 'testsets/ragas_testset_50.csv'
OUTPUT_CSV = 'testset_with_clean_retrieval.csv'
TOP_K = 3
# ---------------------

def clean_chunk_text(full_content):
    """
    Transforms the raw KB markdown into the clean format:
    Title + \n\nRespuesta\n\n + Body (up to footer)
    """
    try:
        lines = full_content.strip().split('\n')
        if not lines: return full_content

        # 1. Extract and Clean Title (Remove '# ' from the start)
        raw_title_line = lines[0].strip()
        clean_title = raw_title_line.lstrip('#').strip()

        # 2. Find the "Respuesta" header (Case insensitive, handles # or ##)
        # We look for the header position to chop off the Metadata above it
        match = re.search(r'(?:#+)?\s*Respuesta', full_content, re.IGNORECASE)
        
        if not match:
            # Fallback: if we can't find 'Respuesta', return text as is (or handle as error)
            return full_content

        # Get text starting immediately after "Respuesta"
        text_after_header = full_content[match.end():]

        # 3. Remove the Footer
        # We stop reading at the horizontal rule '---'
        body_content = text_after_header.split('---')[0].strip()

        # 4. Construct the Final String
        # Format: Title + \n\nRespuesta\n\n + Body
        final_string = f"{clean_title}\n\nRespuesta\n\n{body_content}"
        
        return final_string

    except Exception:
        # If parsing fails fantastically, return original so we don't lose data
        return full_content

def get_retrieved_contexts(client, query):
    """
    Queries Bedrock KB, cleans the results, and returns a list of strings.
    """
    try:
        response = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {'numberOfResults': TOP_K}
            }
        )
        
        clean_chunks = []
        if 'retrievalResults' in response:
            for item in response['retrievalResults']:
                raw_text = item['content']['text']
                # Apply the cleaning logic
                cleaned_text = clean_chunk_text(raw_text)
                clean_chunks.append(cleaned_text)
        
        return clean_chunks

    except Exception as e:
        print(f"Error retrieving for query: {e}")
        return []

def main():
    # 1. Setup Session
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client('bedrock-agent-runtime')
    
    print(f"--- Loading dataset: {INPUT_CSV} ---")
    
    # 2. Read CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("❌ Error: Input CSV file not found.")
        return

    print(f"Loaded {len(df)} rows. Starting retrieval...")

    # 3. Iterate and Retrieve
    retrieved_contexts_column = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        query = row['user_input']
        
        # Get the list of CLEANED strings
        chunks_list = get_retrieved_contexts(client, query)
        
        retrieved_contexts_column.append(chunks_list)

    # 4. Add new column
    df['retrieved_contexts'] = retrieved_contexts_column

    # 5. Save to new CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"\n✅ Success! Saved to: {OUTPUT_CSV}")
    
    # 6. Preview to verify formatting
    print("\n--- Formatting Check (First Item) ---")
    first_list = df['retrieved_contexts'].iloc[0]
    if first_list:
        print(first_list[0])
    else:
        print("No results found for first item.")

if __name__ == "__main__":
    main()