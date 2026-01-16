from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,    
)
from ragas.testset.persona import Persona
import boto3

boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-2')

config = {
    "llm": "openai.gpt-oss-120b-1:0",
    "embeddings": "amazon.titan-embed-text-v2:0",  
    "temperature": 0,
}

FOLDER_PATH = "./old_knowledge_base_small"
OUTPUT_FILE = "testsets/smalltest.csv"
TESTSET_SIZE = 1


sequential_config = RunConfig(
    max_workers=1,
    timeout=60,     # seconds to wait per call
    max_retries=3   
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

loader = DirectoryLoader(
    FOLDER_PATH, 
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")



# CREATE PERSONAS
common_rules = """
IMPORTANTE: Hablas EXCLUSIVAMENTE en Español neutral/chileno.
Debes tomar el rol de un USUARIO, una persona real. NO sabes que estás preguntándole a una base de conocimiento, sino que a un asesor IA.
NUNCA hagas referencia al nombre del archivo, al 'documento', 'texto', 'contexto' o 'información provista'. 
Tipos INCORRECTOS de preguntas que NUNCA debes hacer:
1-"¿Qué dice el BD1-00594 sobre XXXX?" -> MAL, menciona el nombre documento.
3-"Según el documento, ¿cuál es XXXXX?" -> MAL, menciona el documento.

Tus preguntas deben ser NATURALES e 'ingenuas': el usuario no conoce el contenido de los documentos, por lo que sus preguntas no refieren al contenido específico del documento.

Ejemplos de preguntas bien formuladas vs mal formuladas:
1-CORRECTO: cómo contrato una cuenta de ahorro vivienda? -> BIEN
1-INCORRECTO: ¿Cómo puedo contratar una Cuenta de Ahorro Vivienda en Banco Estado y cuáles son todos los requisitos que debo cumplir, como ser persona natural, edad mínima, ausencia de otra cuenta similar y el depósito de apertura requerido? -> MAL, demasiado específico y larga, un usuario no haría esta pregunta.

2-CORRECTO: qué es un crédito hipotecario y cómo funciona? -> BIEN
2-INCORRECTO: Según el documento BD1-00594, ¿qué es un crédito hipotecario y cómo funciona en detalle, incluyendo los diferentes tipos disponibles, las tasas de interés aplicables y los requisitos para solicitarlo? -> MAL, menciona el nombre del documento y es demasiado detallada.

3-CORRECTO: cómo abrir una cuenta de ahorro vivienda y cuáles son los requisitos? -> BIEN
3-INCORRECTO: ¿Cómo puedo abrir la cuenta de ahoro vivienda y qué tengo que cumplir como requisitos de apertura según la guía en línea del banco? -> MAL, demasiado específica y larga, un usuario no haría esta pregunta.

4-CORRECTO: cuál es el deposito de apertura minimo para CAV y y tiene que ver con ahorro minimo para postular al subsidio habitacional? -> BIEN
4-INCORRECTO: Oye, dime cual es el deposito de apertura minimo de UF 0,5 para la cuenta de ahorro vivienda y como se relaciona con el ahorro minimo exigido que tengo que cumplir pa' postular al subsidio habitacional, gracias -> MAL, demasiado larga.

Tu tarea es generar SÓLO preguntas CORRECTAS y BIEN FORMULADAS, siguiendo los ejemplos y reglas anteriores, y el rol de usuario externo.
"""

persona_first_buyer = Persona(
    name="Joven Profesional Primeriza",
    role_description=f"Eres un joven profesional chileno buscando su primer departamento."
                     f"Estás buscando comprar tu primer departamento pero tienes muchas dudas básicas."
                     f"No entiendes bien los términos financieros y preguntas cosas simples sobre créditos hipotecarios, tasas de interés, subsidios, etc."
                     f"{common_rules}"
)

persona_family_investor = Persona(
    name="Padre de Familia Pragmático",
    role_description=f"Eres un padre de familia chileno enfocado en la seguridad y los costos."
                     f"Preguntas directo al grano sobre dividendos, seguros, tasas, etc."
                     f"{common_rules}"
)

persona_learner = Persona(
    name="Estudiante Curioso",
    role_description=f"Eres un estudiante chileno aprendiendo finanzas. Haces preguntas teóricas "
                     f"sobre cómo funciona la inflación, la UF, los créditos, etc. "
                     f"No usas puntuación correcta."
                     f"{common_rules}"
)

persona_small_investor = Persona(
    name="Pequeña Inversionista",
    role_description=f"Adulto de 35 años que ha logrado ahorrar dinero."
                     f"No quieres vivir en la propiedad, sino comprar un departamento pequeño para arrendarlo y mejorar tu jubilación futura. "
                     f"Tus preguntas se enfocan en la rentabilidad, el porcentaje de financiamiento que da el banco para segundas viviendas, los beneficios tributarios, etc. "
                     f"{common_rules}"
)

persona_senior = Persona(
    name="Usuario Senior",
    role_description=f"Un administrativo de 58 años próximo a jubilarse. "
                     f"Escribes mal y haces preguntas confusas, mal escritas, ya que no eres experto en tecnología ni finanzas. "
                     f"{common_rules}"
)


personas = [persona_first_buyer, persona_family_investor, persona_learner, persona_small_investor, persona_senior]

# QUERY DISTRIBUTION
syn_single = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
syn_multi_spec = MultiHopSpecificQuerySynthesizer(llm=generator_llm)
syn_multi_abs = MultiHopAbstractQuerySynthesizer(llm=generator_llm)

distributions = [
    (syn_single, 0.8),
    (syn_multi_spec, 0.1),
    (syn_multi_abs, 0.1)
]

# RUN THE GENERATION


generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings, 
    persona_list=personas)

dataset = generator.generate_with_langchain_docs(
    documents, 
    testset_size=TESTSET_SIZE,
    run_config=sequential_config, 
    query_distribution=distributions,  
)

df = dataset.to_pandas()

output_filename = OUTPUT_FILE
df.to_csv(output_filename, index=False)

print(f"Success! Testset saved to {OUTPUT_FILE}")

['Contexto erróneo para test']