from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
import boto3

boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-2')

config = {
    # "llm": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "llm": "openai.gpt-oss-120b-1:0",
    "embeddings": "amazon.titan-embed-text-v2:0",  
    "temperature": 0,
}

sequential_config = RunConfig(
    max_workers=1,  # This forces one task at a time
    timeout=60,     # Optional: seconds to wait per call
    max_retries=3   # Optional: retries per failure
)

generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
    client=boto3_bedrock,
    model=config["llm"],
    temperature=config["temperature"],
    max_tokens=4000,
))

generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    model_id=config["embeddings"],
))

# ####################
# LOAD MARKDOWN FILES

# folder_path = "./knowledge_base" 
folder_path = "./knowledge_base_small" 
loader = DirectoryLoader(
    folder_path, 
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader # Optional: specifically uses Markdown loader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")



generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(
    documents, 
    testset_size=5,
    run_config=sequential_config
)

df = dataset.to_pandas()

output_filename = "ragas_testset.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Success! Testset saved to {output_filename}")