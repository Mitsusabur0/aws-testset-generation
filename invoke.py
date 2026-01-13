import boto3
import logging

from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invoke_agent(client, agent_id, alias_id, prompt, session_id):
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            enableTrace=True,
            sessionId = session_id,
            inputText=prompt,
            streamingConfigurations = { 
                "applyGuardrailInterval" : 20,
                "streamFinalResponse" : False
            }
        )
        completion = ""
        for event in response.get("completion"):
            #Collect agent output.
            if 'chunk' in event:
                chunk = event["chunk"]
                completion += chunk["bytes"].decode()
            
            # Log trace output.
            if 'trace' in event:
                trace_event = event.get("trace")
                trace = trace_event['trace']
                for key, value in trace.items():
                    logging.info("%s: %s",key,value)

        print(f"Agent response: {completion}")


if __name__ == "__main__":

    client=boto3.client(
            service_name="bedrock-agent-runtime"
        )
    
    agent_id = "ADTXVTUDWQ"
    alias_id = "A32TZLSKTR"
    session_id = "test1"
    prompt = "Qué es el IVA?"

    try:

        invoke_agent(client, agent_id, alias_id, prompt, session_id)

    except ClientError as e:
        print(f"Client error: {str(e)}")
        logger.error("Client error: %s", {str(e)})