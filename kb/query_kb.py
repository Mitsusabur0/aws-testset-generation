import boto3
import json
import re

# --- CONFIGURATION ---
PROFILE = 'sandbox'
REGION = 'us-east-1'
KB_ID = '3TPM53DPBN' 
OUTPUT_FILE = 'extracted_answers_complete.json'

QUERY_TEXT = "Cuales son los beneficios de fogaes?"
NUMBER_OF_RESULTS = 3
# ---------------------

def clean_extracted_text(full_content):
    """
    1. Extracts the Title (Line 1).
    2. Extracts the 'Respuesta' body.
    3. Combines them as: Title + \n\n + Body
    """
    try:
        lines = full_content.strip().split('\n')
        if not lines: return None

        # 1. Capture the Raw Title
        raw_title_line = lines[0].strip()
        title_text_only = raw_title_line.lstrip('#').strip()

        # 2. Find start of "Respuesta"
        start_match = re.search(r'(?:#+)\s*Respuesta', full_content, re.IGNORECASE)
        if not start_match:
            return "SECTION_MISSING"

        start_index = start_match.end()
        text_after_header = full_content[start_index:]

        # 3. Cut off at the footer separator '---'
        content_block = text_after_header.split('---')[0]
        clean_block = content_block.strip()

        # 4. Remove the repeated title at the end
        if clean_block.endswith(title_text_only):
            clean_block = clean_block[:-len(title_text_only)].strip()
            
        # 5. Combine: Title + \n\n + Cleaned Body
        final_output = f"{raw_title_line}\n\n{clean_block}"
        
        return final_output

    except Exception as e:
        return f"PARSING_ERROR: {str(e)}"

def run_extraction():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client('bedrock-agent-runtime')

    print(f"--- Querying KB: {KB_ID} ---")
    
    response = client.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={'text': QUERY_TEXT},
        retrievalConfiguration={
            'vectorSearchConfiguration': {'numberOfResults': NUMBER_OF_RESULTS}
        }
    )

    results = []
    
    for item in response.get('retrievalResults', []):
        raw_text = item['content']['text']
        
        # Run the extraction logic
        final_text = clean_extracted_text(raw_text)
        
        # Get Source URI
        uri = "Unknown"
        if 's3Location' in item['location']:
            uri = item['location']['s3Location']['uri']

        results.append({
            "score": item['score'],
            "source_uri": uri,
            "final_content": final_text,
            "raw_text_snippet": raw_text  # <--- INCLUDED AS REQUESTED
        })

    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Processed {len(results)} items.")
    print(f"✅ Saved to: {OUTPUT_FILE}")
    
    if results:
        print("\n--- Preview of First Item ---")
        print(f"Cleaned:\n{results[0]['final_content'][:100]}...")

if __name__ == "__main__":
    run_extraction()