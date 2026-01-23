import boto3
import json

# ============================================
# EDIT THESE VARIABLES
# ============================================
AGENT_ID = "UKQEMRZQUS"  # RAG
AGENT_ALIAS_ID = "TP5NTBGTFE"  



SESSION_ID = "test-session-123"  # You can use any unique identifier
USER_INPUT = "Qué es fogaes?"
AWS_REGION = "us-east-1"  # Change to your region
AWS_PROFILE = "sandbox"  # AWS credentials profile

# ============================================
# Initialize Bedrock Agent Runtime Client
# ============================================
session = boto3.Session(profile_name=AWS_PROFILE)
client = session.client(
    'bedrock-agent-runtime',
    region_name=AWS_REGION
)

# ============================================
# Invoke Agent with enableTrace=True
# ============================================
response = client.invoke_agent(
    agentId=AGENT_ID,
    agentAliasId=AGENT_ALIAS_ID,
    sessionId=SESSION_ID,
    inputText=USER_INPUT,
    enableTrace=True  # This is crucial to see retrieved context
)

# ============================================
# Process the Response Stream
# ============================================
print("=" * 60)
print("AGENT RESPONSE AND RETRIEVED CONTEXT")
print("=" * 60)

event_stream = response['completion']
full_response = ""
citations = []

for event in event_stream:
    
    # Agent's text response
    if 'chunk' in event:
        chunk = event['chunk']
        if 'bytes' in chunk:
            text = chunk['bytes'].decode('utf-8')
            full_response += text
            print(f"\nAgent Response: {text}")
    
    # Trace events - this contains retrieved context
    if 'trace' in event:
        trace = event['trace'].get('trace', {})
        
        # Knowledge Base Lookup trace
        if 'orchestrationTrace' in trace:
            orch_trace = trace['orchestrationTrace']
            
            # Retrieved context from Knowledge Base
            if 'observation' in orch_trace:
                observation = orch_trace['observation']
                
                if 'knowledgeBaseLookupOutput' in observation:
                    kb_output = observation['knowledgeBaseLookupOutput']
                    
                    print("\n" + "=" * 60)
                    print("RETRIEVED CONTEXT FROM KNOWLEDGE BASE")
                    print("=" * 60)
                    
                    retrieved_references = kb_output.get('retrievedReferences', [])
                    
                    for idx, ref in enumerate(retrieved_references, 1):
                        print(f"\n--- Reference {idx} ---")
                        
                        # Content of the retrieved chunk
                        content = ref.get('content', {}).get('text', 'N/A')
                        print(f"Content: {content}")
                        
                        # Source location
                        location = ref.get('location', {})
                        if 's3Location' in location:
                            s3_loc = location['s3Location']
                            print(f"Source: s3://{s3_loc.get('uri', 'N/A')}")
                        
                        # Relevance score
                        metadata = ref.get('metadata', {})
                        score = metadata.get('score')
                        if score:
                            print(f"Relevance Score: {score}")
                        
                        print("-" * 40)
            
            # Model invocation input (what was sent to the LLM)
            if 'modelInvocationInput' in orch_trace:
                model_input = orch_trace['modelInvocationInput']
                print("\n" + "=" * 60)
                print("PROMPT SENT TO MODEL (with context)")
                print("=" * 60)
                print(json.dumps(model_input, indent=2))
    
    # Citations (if available)
    if 'chunk' in event:
        chunk = event['chunk']
        if 'attribution' in chunk:
            attribution = chunk['attribution']
            if 'citations' in attribution:
                citations.extend(attribution['citations'])

# ============================================
# Display Final Response and Citations
# ============================================
print("\n" + "=" * 60)
print("FINAL AGENT RESPONSE")
print("=" * 60)
print(full_response)

if citations:
    print("\n" + "=" * 60)
    print("CITATIONS")
    print("=" * 60)
    for idx, citation in enumerate(citations, 1):
        print(f"\nCitation {idx}:")
        print(json.dumps(citation, indent=2))

print("\n" + "=" * 60)
print("SESSION COMPLETE")
print("=" * 60)