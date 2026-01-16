import os
import boto3
import json
import pprint
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_core.documents import Document
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.cost import TokenUsage

# --- GLOBAL DEBUG COUNTER ---
# We use this to verify counts even if Ragas says 0
DEBUG_STATS = {
    "calls": 0,
    "input_found": 0,
    "output_found": 0
}

def debug_token_parser(llm_result):
    """
    A verbose parser that prints the internal structure of the LLM result
    so we can find where the token usage data is hiding.
    """
    DEBUG_STATS["calls"] += 1
    print(f"\n[PARSER] Call #{DEBUG_STATS['calls']} triggered")
    
    input_tokens = 0
    output_tokens = 0
    
    # 1. Check Top-Level LLM Output (Sometimes aggregated here)
    if hasattr(llm_result, 'llm_output') and llm_result.llm_output:
        print(f"  [DEBUG] Found llm_output: {llm_result.llm_output}")
        # Bedrock sometimes puts it here under 'usage'
        if 'usage' in llm_result.llm_output:
            u = llm_result.llm_output['usage']
            # Handle both CamelCase (AWS) and snake_case
            i = u.get('inputTokens', u.get('input_tokens', u.get('prompt_tokens', 0)))
            o = u.get('outputTokens', u.get('output_tokens', u.get('completion_tokens', 0)))
            if i > 0 or o > 0:
                print(f"  [SUCCESS] Found tokens in llm_output: In={i}, Out={o}")
                input_tokens += i
                output_tokens += o

    # 2. Iterate through Generations (Per-message metadata)
    for i, generations in enumerate(llm_result.generations):
        for j, gen in enumerate(generations):
            # print(f"  [DEBUG] Inspecting Gen {i}.{j} type: {type(gen)}")
            
            # CHECK A: gen.message.response_metadata (Chat Models standard)
            if hasattr(gen, 'message'):
                # print(f"    -> Has 'message' attribute")
                if hasattr(gen.message, 'response_metadata'):
                    meta = gen.message.response_metadata
                    # print(f"    -> response_metadata keys: {meta.keys()}")
                    if 'usage' in meta:
                        u = meta['usage']
                        print(f"    [SUCCESS] Found usage in message.response_metadata: {u}")
                        input_tokens += u.get('inputTokens', 0)
                        output_tokens += u.get('outputTokens', 0)
            
            # CHECK B: gen.generation_info (Legacy/Standard LLM)
            if hasattr(gen, 'generation_info') and gen.generation_info:
                # print(f"    -> Has 'generation_info': {gen.generation_info.keys()}")
                if 'usage' in gen.generation_info:
                    u = gen.generation_info['usage']
                    print(f"    [SUCCESS] Found usage in generation_info: {u}")
                    input_tokens += u.get('inputTokens', 0)
                    output_tokens += u.get('outputTokens', 0)

    # Update Global Stats
    DEBUG_STATS["input_found"] += input_tokens
    DEBUG_STATS["output_found"] += output_tokens
    
    print(f"  [PARSER END] Total found this call: {input_tokens} in / {output_tokens} out")
    
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

# --- SETUP ---
boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-2')

config = {
    "llm": "openai.gpt-oss-120b-1:0", 
    "embeddings": "amazon.titan-embed-text-v2:0",  
}

# NOTE: max_retries=0 ensures we fail fast if there is an error
run_config = RunConfig(max_workers=1, max_retries=0) 

generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
    client=boto3_bedrock,
    model=config["llm"],
    max_tokens=2000,
))

generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    client=boto3_bedrock,
    model_id=config["embeddings"],
))

documents = [
    Document(
        page_content="""
        El BancoEstado ofrece la 'Cuenta de Ahorro Vivienda'. 
        Esta cuenta permite postular al Subsidio Habitacional.
        Requisitos: Ser persona natural mayor de 18 años.
        El BancoEstado ofrece la 'Cuenta de Ahorro Vivienda'. 
        Esta cuenta permite postular al Subsidio Habitacional.
        Requisitos: Ser persona natural mayor de 18 años.
        El BancoEstado ofrece la 'Cuenta de Ahorro Vivienda'. 
        Esta cuenta permite postular al Subsidio Habitacional.
        Requisitos: Ser persona natural mayor de 18 años.
        El BancoEstado ofrece la 'Cuenta de Ahorro Vivienda'. 
        Esta cuenta permite postular al Subsidio Habitacional.
        Requisitos: Ser persona natural mayor de 18 años.
        """,
        metadata={"filename": "guia_banco.md"}
    )
]

# --- RUN ---
print(">>> STARTING GENERATION WITH DEBUG PARSER <<<")

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings
)

try:
    dataset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=1, 
        run_config=run_config,
        token_usage_parser=debug_token_parser 
    )
    print("\n>>> GENERATION FINISHED <<<")

except Exception as e:
    print(f"\n[ERROR] Generation failed: {e}")

# --- REPORT ---
print("\n" + "="*40)
print("FINAL TOKEN REPORT (FROM GLOBAL STATS)")
print("="*40)
print(f"Total LLM Calls: {DEBUG_STATS['calls']}")
print(f"Input Tokens:    {DEBUG_STATS['input_found']}")
print(f"Output Tokens:   {DEBUG_STATS['output_found']}")
print("="*40)

# Calculate cost manually
cost = (DEBUG_STATS['input_found'] * 3.0 / 1e6) + (DEBUG_STATS['output_found'] * 15.0 / 1e6)
print(f"Manual Cost Calculation: ${cost:.6f}")

# Save to JSON
with open("tokens.json", "w") as f:
    json.dump(DEBUG_STATS, f, indent=4)
print("Saved debug stats to tokens.json")