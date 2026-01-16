import os
import glob
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import (
    StylingConfig, 
    EvolutionConfig, 
    ContextConstructionConfig
)

# 1. Load Environment Variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

def generate_chilean_bank_testset():
    print("--- Starting Synthetic Data Generation PoC ---")

    # 2. Define File Paths
    # Assuming files are in a folder named 'knowledge_base'
    kb_folder = "old_knowledge_base_small"
    document_paths = glob.glob(os.path.join(kb_folder, "*.md"))
    
    if not document_paths:
        raise FileNotFoundError(f"No .md files found in {kb_folder}. Please add your 5 files.")
    
    print(f"Found {len(document_paths)} documents.")

    # --- DEBUG STEP: Verify files are readable ---
    # This prevents the "0 out of 0 chunks" error by catching encoding issues early.
    print("Verifying file readability...")
    for path in document_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    print(f"[WARNING] File is empty: {path}")
        except Exception as e:
            print(f"[ERROR] Could not read {path}. Rename the file to remove special characters (accents). Error: {e}")
            return # Stop execution if files are unreadable
    # ---------------------------------------------

    # 3. Configure Styling (Language & Currency Context)
    styling_config = StylingConfig(
        input_format="Preguntas en español. Tomar el rol de un usuario promedio chileno. Las preguntas deben ser realistas. Un usuario pregunta cosas generales, no específicas. El usuario no tiene conocimiento técnico. El usuario no conoce qué se requiere, por lo que no menciona lo que tiene actualmente. Sus preguntas son conversacionales e informales, no tan correctas",
        expected_output_format="Respuestas claras y útiles en español. NUNCA se debe responder más de lo que se encuentra en el documento. Las respuestas SÓLO pueden incluir el contenido del documento.",
        task="Asistente virtual para un banco chileno especializado en créditos hipotecarios y educación financiera.",
        scenario="Un cliente chileno está haciendo preguntas sobre productos bancarios, tasas, o consejos financieros. Las preguntas las responde el documento, pero no preguntan específicamente por detalles del documento. EL USUARIO NO HA LEÍDO EL DOCUMENTO, POR LO QUE NO PUEDE PREGUNTAR POR DETALLES TÉCNICOS O ESPECÍFICOS DEL DOCUMENTO.",
    )

    # 4. Configure Evolution (RAG-Safe)
    # We avoid 'REASONING' or 'HYPOTHETICAL' to prevent hallucinations outside the KB.
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.MULTICONTEXT: 0.25,
            Evolution.CONCRETIZING: 0.25,
            Evolution.CONSTRAINED: 0.25,
            Evolution.COMPARATIVE: 0.25
        },
        num_evolutions=1  # Default complexity
    )

    # 5. Configure Context Construction (NO CHUNKING)
    # We use a large chunk_size (5000) to ensure the whole file is treated as 1 chunk.
    # encoding='utf-8' ensures we don't crash on Spanish accents.
    context_construction_config = ContextConstructionConfig(
        max_contexts_per_document=1,
        chunk_size=5000, # Large value prevents splitting the file
        chunk_overlap=0,
        encoding="utf-8"
    )

    # 6. Initialize Synthesizer
    synthesizer = Synthesizer(
        model="gpt-5-nano", 
        styling_config=styling_config,
        evolution_config=evolution_config
    )

    # 7. Generate Goldens
    print(f"Generating goldens from {len(document_paths)} documents...")
    
    generated_goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        context_construction_config=context_construction_config,
        include_expected_output=True,
        max_goldens_per_context=2
    )

    # Safety check before saving
    if not generated_goldens:
        print("[ERROR] No goldens were generated. Please check the logs above for '0 out of 0 chunks'.")
        return

    # 8. Save Output
    output_filename = "chilean_bank_goldens"
    print(f"Generation complete. Saving to {output_filename}.json and .csv...")
    
    # Save as JSON (Standard DeepEval format)
    synthesizer.save_as(
        file_type='json',
        directory="./synthetic_data",
        file_name=output_filename
    )
    
    # Save as CSV (Easier for non-technical stakeholders to review in Excel)
    synthesizer.save_as(
        file_type='csv',
        directory="./synthetic_data",
        file_name=output_filename
    )

    # Optional: Convert to Pandas to show a preview
    df = synthesizer.to_pandas()
    print("\nPreview of Generated Data:")
    print(df[['input', 'expected_output']].head())

if __name__ == "__main__":
    generate_chilean_bank_testset()