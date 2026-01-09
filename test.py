from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import boto3

boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-2')

config = {
    "llm": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "embeddings": "amazon.titan-embed-text-v2:0",  
    "temperature": 0.4,
}

generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
    client=boto3_bedrock,
    model=config["llm"],
    temperature=config["temperature"],
    max_tokens=100,
))

generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    model_id=config["embeddings"],
))

# ####################
# LOAD MARKDOWN FILES

folder_path = "./knowledge_base" 
loader = DirectoryLoader(
    folder_path, 
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader # Optional: specifically uses Markdown loader
)

documents = loader.load()
print(f"Loaded {len(documents)} documents.")
