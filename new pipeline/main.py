import os
import json
import random
import boto3
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm # You may need to pip install tqdm

from langchain_aws import ChatBedrockConverse
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ==========================================
# CONFIGURATION
# ==========================================

FOLDER_PATH = "./kb_nuevo_pipeline"
OUTPUT_FILE = "testsets/test_nuevo_pipeline_manual.csv"
TESTSET_SIZE = 30 # Number of *successful* samples desired
MAX_RETRIES = 3   # How many times to retry generating if the Critic rejects

# BEDROCK CONFIG
# Note: Ensure "openai.gpt-oss-120b-1:0" is the correct ID for your Bedrock Setup. 
# Usually Bedrock IDs look like "anthropic.claude-3-sonnet-..." or "amazon.titan..."
BEDROCK_MODEL_ID = "openai.gpt-oss-120b-1:0" 
REGION_NAME = "us-east-2"

# ==========================================
# PERSONAS DEFINITION
# ==========================================

COMMON_RULES = """
- Tu idioma es Español de Chile.
- Eres un USUARIO, no un experto.
- NUNCA menciones "el texto", "el PDF", "el documento provisto" o códigos internos.
- Tus preguntas deben ser naturales, breves y directas.
- No uses ortografía perfecta si el rol indica lo contrario.
"""

PERSONAS = [
    {
        "name": "Joven Profesional Primeriza",
        "desc": f"Joven chilena buscando su primer depa. No entiende términos financieros complejos. Pregunta simple. {COMMON_RULES}"
    },
    {
        "name": "Padre de Familia Pragmático",
        "desc": f"Padre chileno, preocupado por costos y seguridad. Va directo al grano. {COMMON_RULES}"
    },
    {
        "name": "Estudiante Curioso",
        "desc": f"Estudiante universitario. Pregunta cosas teóricas (UF, inflación). Escribe casual, a veces sin signos de interrogación. {COMMON_RULES}"
    },
    {
        "name": "Pequeña Inversionista",
        "desc": f"Mujer de 35 años comprando para arrendar. Pregunta por rentabilidad y financiamiento. {COMMON_RULES}"
    },
    {
        "name": "Usuario Senior",
        "desc": f"Hombre de 58 años, poco tecnológico. Preguntas a veces confusas o mal redactadas. {COMMON_RULES}"
    }
]

# ==========================================
# PROMPTS
# ==========================================

GENERATOR_PROMPT = """
Actúa como: {persona_desc}

Tu Tarea:
Lee el siguiente fragmento de información bancaria y formula una pregunta que TÚ (en tu rol) harías para obtener esta información. Luego redacta la respuesta ideal.

Fragmento:
"{context_text}"

Reglas Estrictas:
1. La pregunta debe sonar natural, dicha por una persona en Chile.
2. NO hagas referencia al texto (ej: "según el fragmento...").
3. La respuesta debe ser veraz y estar contenida 100% en el fragmento.

Formato de Salida (JSON):
{{
    "question": "Tu pregunta aquí",
    "ground_truth": "La respuesta correcta extraída del texto"
}}
"""

CRITIC_PROMPT = """
Actúa como un Auditor de Calidad de Datos (Critic).
Evalúa el siguiente par Pregunta/Respuesta generado a partir de un contexto.

Contexto: "{context_text}"
Pregunta Generada: "{question}"
Respuesta Generada: "{ground_truth}"

Criterios de Aprobación (Deben cumplirse TODOS):
1. La pregunta NO menciona "el texto", "el documento", "la información dada" ni nada meta-referencial.
2. La respuesta es correcta y está totalmente respaldada por el Contexto.
3. La pregunta tiene sentido por sí misma (no depende de leer el contexto previamente).
4. El idioma parece Español Chileno / Natural (no robótico).

Salida (JSON):
{{
    "approved": true/false,
    "reason": "Explica brevemente por qué aprobaste o rechazaste"
}}
"""

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_json_output(content: str) -> dict:
    """Cleans markdown fencing from LLM output and parses JSON."""
    content = content.replace("```json", "").replace("```", "").strip()
    # Sometimes models add extra text, try to find the first { and last }
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end != -1:
        content = content[start:end]
    return json.loads(content)

def init_llm():
    boto3_client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
    return ChatBedrockConverse(
        client=boto3_client,
        model=BEDROCK_MODEL_ID,
        temperature=0.7, # Creativity for questions
        max_tokens=2000,
    )

def load_documents():
    print(f"Loading documents from {FOLDER_PATH}...")
    loader = DirectoryLoader(FOLDER_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()
    
    # Split into meaningful chunks (size is important for specific questions)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} files. Created {len(chunks)} chunks.")
    return chunks

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    llm = init_llm()
    chunks = load_documents()
    
    if not chunks:
        print("Error: No documents found.")
        return

    testset_data = []
    
    # Progress bar loop
    pbar = tqdm(total=TESTSET_SIZE, desc="Generating Testset")
    
    while len(testset_data) < TESTSET_SIZE:
        
        # 1. Select Random Inputs
        chunk = random.choice(chunks)
        persona = random.choice(PERSONAS)
        context_text = chunk.page_content
        
        # Skip very short chunks (usually noise)
        if len(context_text) < 100:
            continue

        # -----------------------
        # STEP 1: GENERATION
        # -----------------------
        try:
            gen_prompt = GENERATOR_PROMPT.format(
                persona_desc=persona["desc"],
                context_text=context_text
            )
            
            response = llm.invoke(gen_prompt)
            gen_data = clean_json_output(response.content)
            
            question = gen_data.get("question")
            ground_truth = gen_data.get("ground_truth")

        except Exception as e:
            # print(f"Generation parsing error: {e}")
            continue # Skip to next attempt

        # -----------------------
        # STEP 2: CRITIC VALIDATION
        # -----------------------
        try:
            critic_input = CRITIC_PROMPT.format(
                context_text=context_text,
                question=question,
                ground_truth=ground_truth
            )
            
            # Use lower temp for critic to be strict
            response_critic = llm.invoke(critic_input) 
            critic_data = clean_json_output(response_critic.content)
            
            is_approved = critic_data.get("approved", False)
            reason = critic_data.get("reason", "No reason provided")

            if is_approved:
                # Success! Add to dataset
                testset_data.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "reference_contexts": [context_text], 
                    "persona": persona["name"],
                    "source": chunk.metadata.get("source", "unknown"),
                    "critic_comment": reason
                })
                pbar.update(1)
            else:
                # Optional: Print why it failed to fine-tune prompts later
                # print(f"Rejected ({persona['name']}): {reason}")
                pass

        except Exception as e:
            # print(f"Critic error: {e}")
            continue

    pbar.close()

    # Save to CSV
    df = pd.DataFrame(testset_data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Generated {len(df)} validated samples.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()