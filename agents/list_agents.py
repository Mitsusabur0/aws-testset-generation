import boto3
import json

# Configuration
PROFILE = 'sandbox'
REGION = 'us-east-1'
OUTPUT_FILE = 'agents.json'

def export_agents():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client('bedrock-agent')
    
    agents_list = []
    
    print(f"Fetching agents from profile '{PROFILE}'...")

    # Use paginator to ensure we get ALL agents, not just the first page
    paginator = client.get_paginator('list_agents')
    
    for page in paginator.paginate():
        for agent in page['agentSummaries']:
            # Create a clean dictionary with only ID and Name
            agent_data = {
                "id": agent['agentId'],
                "name": agent['agentName']
            }
            agents_list.append(agent_data)

    # Save to JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(agents_list, f, indent=4)
    
    print(f"✅ Successfully saved {len(agents_list)} agents to {OUTPUT_FILE}")


import boto3
import json
import time
from botocore.config import Config
from botocore.exceptions import ClientError

# Configuration
PROFILE = 'sandbox'
REGION = 'us-east-1'
INPUT_FILE = 'agents.json'
OUTPUT_FILE = 'agents_with_aliases.json'
TIMEOUT_SECONDS = 10  # Hard limit for connection and read

def get_aliases_for_agent(client, agent_id, agent_name):
    """Fetches aliases for a single agent with error handling."""
    aliases_data = []
    
    try:
        # We use a paginator in case an agent has many aliases
        paginator = client.get_paginator('list_agent_aliases')
        
        # We invoke the API
        for page in paginator.paginate(agentId=agent_id):
            for alias in page['agentAliasSummaries']:
                
                # Extract routing version safely
                routing_ver = "Unknown"
                if 'routingConfiguration' in alias and len(alias['routingConfiguration']) > 0:
                    routing_ver = alias['routingConfiguration'][0]['agentVersion']

                aliases_data.append({
                    "alias_name": alias['agentAliasName'],
                    "alias_id": alias['agentAliasId'],
                    "status": alias['agentAliasStatus'],
                    "routing_to_version": routing_ver
                })
                
    # except (ConnectTimeout, ReadTimeout):
    #     print(f"   ❌ Timeout: Waited {TIMEOUT_SECONDS}s for agent '{agent_name}'. Skipping.")
    #     return {"error": "Timeout"}
    except ClientError as e:
        print(f"   ❌ AWS Error for '{agent_name}': {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"   ❌ Unexpected Error for '{agent_name}': {e}")
        return {"error": str(e)}

    return aliases_data

def process_agents():
    # 1. Setup Boto3 with Timeouts
    my_config = Config(
        connect_timeout=TIMEOUT_SECONDS, 
        read_timeout=TIMEOUT_SECONDS,
        retries={'max_attempts': 1} # Don't retry endlessly if it fails
    )
    
    try:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        client = session.client('bedrock-agent', config=my_config)
    except Exception as e:
        print(f"CRITICAL: Could not create AWS session. Check credentials. {e}")
        return

    # 2. Load input file
    try:
        with open(INPUT_FILE, 'r') as f:
            agents_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run the previous script first.")
        return

    final_output = []
    total = len(agents_list)

    print(f"--- Processing {total} agents from {INPUT_FILE} ---")

    # 3. Iterate through agents
    for index, agent in enumerate(agents_list):
        agent_id = agent['id']
        agent_name = agent['name']
        
        print(f"[{index+1}/{total}] Fetching aliases for: {agent_name} ({agent_id})...")
        
        # Get aliases
        aliases = get_aliases_for_agent(client, agent_id, agent_name)
        
        # Build the new object structure
        agent_record = {
            "agent_name": agent_name,
            "agent_id": agent_id,
            "aliases": aliases
        }
        
        final_output.append(agent_record)

    # 4. Save to new JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"\n✅ Done! Results saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    export_agents()
    process_agents()