from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.synthesizers import (
    SpecificQuerySynthesizer,
    AbstractQuerySynthesizer,
    ComparativeAbstractQuerySynthesizer,
)
from ragas.testset.persona import Persona

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



# CREATE PERSONAS
common_rules = """
IMPORTANTE: Hablas EXCLUSIVAMENTE en Español neutral/chileno.
NUNCA hagas referencia al nombre del archivo, al 'documento', 'texto', 'contexto' o 'información provista'. 
NUNCA debes mencionar el documento, NUNCA preguntar: "según BD1-00001?". 
Debes tomar el rol de un USUARIO, una persona real. NO sabes que estás preguntándole a una base de conocimiento, sino que a una IA.
Ejemplo MALO: ¿Qué dice el BD1-00594 sobre XXXX?" 
Ejemplo MALO: "Oye, y con ese BD1-00599, ¿qué nacionalidad tengo que tener pa' pedir crédito hipotecario?".
Ejemplo MALO: "Según el documento, ¿cuál es XXXXX?".

"""

persona_first_buyer = Persona(
    name="Comprador Primera Vivienda",
    role_description=f"Eres un joven profesional chileno buscando su primer departamento. "
                     f"No entiendes términos financieros complejos. Preguntas con dudas básicas. "
                     f"{common_rules}"
)

persona_family_investor = Persona(
    name="Padre de Familia Pragmático",
    role_description=f"Eres un padre de familia chileno enfocado en la seguridad y los costos. "
                     f"Preguntas directo al grano sobre dividendos, seguros y tasas. "
                     f"{common_rules}"
)

persona_learner = Persona(
    name="Estudiante Curioso",
    role_description=f"Eres un estudiante chileno aprendiendo finanzas. Haces preguntas teóricas "
                     f"sobre cómo funciona la inflación, la UF y los créditos. "
                     f"{common_rules}"
)

personas = [persona_first_buyer, persona_family_investor, persona_learner]


# QUERY DISTRIBUTION
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]



# RUN THE GENERATION


generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings, 
    persona_list=personas)

dataset = generator.generate_with_langchain_docs(
    documents, 
    testset_size=5,
    run_config=sequential_config,   
)

df = dataset.to_pandas()

output_filename = "ragas_testset.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Success! Testset saved to {output_filename}")