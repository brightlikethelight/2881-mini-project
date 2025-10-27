#!/usr/bin/env python3
"""
Create test Spanish Wikipedia data from sample articles.
This is a temporary solution until full dump is downloaded.
"""

import json
from pathlib import Path
from datetime import datetime

# Sample Spanish Wikipedia articles for testing
SPANISH_ARTICLES = [
    {
        "title": "Inteligencia artificial",
        "text": """
La inteligencia artificial (IA) es la disciplina de la ciencia de la computación que se dedica a crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas pueden incluir capacidades como el reconocimiento de patrones, la toma de decisiones y el procesamiento del lenguaje natural.
""",
        "create_timestamp": "2023-11-15T10:30:00Z"
    },
    {
        "title": "Python (lenguaje de programación)",
        "text": """
Python es un lenguaje de programación de alto nivel, interpretado y de propósito general. Fue diseñado por Guido van Rossum y lanzado por primera vez en 1991. Python se destaca por su sintaxis clara y legible, lo que lo hace ideal para principiantes y desarrollo rápido de aplicaciones.
""",
        "create_timestamp": "2023-11-20T14:15:00Z"
    },
    {
        "title": "Machine Learning",
        "text": """
El aprendizaje automático o machine learning es un subcampo de la inteligencia artificial que se centra en el diseño de algoritmos que pueden aprender de los datos y hacer predicciones o decisiones sin ser programados explícitamente para cada tarea.
""",
        "create_timestamp": "2023-11-25T09:00:00Z"
    },
    {
        "title": "Deep Learning",
        "text": """
El aprendizaje profundo o deep learning es una rama del aprendizaje automático que utiliza redes neuronales artificiales con múltiples capas para modelar y entender patrones complejos en datos. Estas redes profundas han logrado avances significativos en visión por computadora, procesamiento de lenguaje natural y reconocimiento de voz.
""",
        "create_timestamp": "2023-11-28T16:45:00Z"
    },
    {
        "title": "Transformadores (arquitectura neuronal)",
        "text": """
Los transformadores son una arquitectura de redes neuronales diseñada para procesar secuencias de datos, especialmente texto. Introducida en 2017, esta arquitectura utiliza mecanismos de atención para capturar relaciones entre elementos distantes en secuencias. Los transformadores han revolucionado el procesamiento de lenguaje natural y son la base de modelos como GPT, BERT y T5.
""",
        "create_timestamp": "2023-12-01T08:20:00Z"
    },
    {
        "title": "BERT (modelo de lenguaje)",
        "text": """
BERT (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje basado en la arquitectura de transformadores desarrollado por Google. A diferencia de los modelos anteriores que procesaban texto secuencialmente, BERT utiliza codificación bidireccional para entender el contexto completo de cada palabra tanto a la izquierda como a la derecha.
""",
        "create_timestamp": "2023-12-05T11:30:00Z"
    },
    {
        "title": "GPT (Generative Pre-trained Transformer)",
        "text": """
GPT es una serie de modelos de lenguaje basados en la arquitectura de transformadores desarrollados por OpenAI. GPT utiliza un enfoque de aprendizaje no supervisado con predicción del siguiente token. La serie incluye GPT-2, GPT-3, GPT-4, y modelos posteriores que han demostrado capacidades impresionantes en generación de texto, comprensión de lenguaje y tareas de razonamiento.
""",
        "create_timestamp": "2023-12-10T13:15:00Z"
    },
    {
        "title": "Procesamiento de lenguaje natural",
        "text": """
El procesamiento de lenguaje natural (NLP) es una rama de la inteligencia artificial que se enfoca en la interacción entre computadoras y lenguaje humano. Las tareas incluyen análisis de sentimientos, traducción automática, reconocimiento de entidades nombradas, respuesta a preguntas y generación de texto.
""",
        "create_timestamp": "2023-12-15T10:00:00Z"
    },
    {
        "title": "RAG (Retrieval-Augmented Generation)",
        "text": """
RAG, o generación aumentada por recuperación, es un paradigma de inteligencia artificial que combina la recuperación de información con la generación de texto. Primero se recuperan documentos relevantes de una base de conocimiento externa, y luego se utilizan estos documentos como contexto para generar respuestas más precisas y fundamentadas.
""",
        "create_timestamp": "2023-12-20T15:30:00Z"
    },
    {
        "title": "Vectorización de texto",
        "text": """
La vectorización de texto es el proceso de convertir palabras o frases en representaciones numéricas (vectores) que capturan su significado semántico. Técnicas comunes incluyen word embeddings como Word2Vec, GloVe y más recientemente, modelos basados en transformadores como BERT y GPT que generan vectores contextuales.
""",
        "create_timestamp": "2023-12-25T12:00:00Z"
    }
]

def create_test_spanish_data():
    """Create test Spanish Wikipedia dataset."""
    output_dir = Path("raw_data/private/wiki_spanish")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating test Spanish Wikipedia dataset...")
    print(f"Location: {output_dir}")
    
    # Save individual article files
    for i, article in enumerate(SPANISH_ARTICLES):
        article_file = output_dir / f"article_{i:04d}.txt"
        with open(article_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {article['title']}\n\n")
            f.write(article['text'])
    
    # Save metadata
    metadata = {
        'source_dump': 'synthetic_test_data',
        'dump_filename': 'test_spanish_articles.json',
        'collection_date': datetime.now().isoformat(),
        'num_articles': len(SPANISH_ARTICLES),
        'language': 'es',
        'articles': [
            {
                'id': i,
                'title': a['title'],
                'create_timestamp': a['create_timestamp'],
            }
            for i, a in enumerate(SPANISH_ARTICLES)
        ]
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created {len(SPANISH_ARTICLES)} Spanish articles")
    print(f"  Saved to: {output_dir.absolute()}")
    print(f"\nNext steps:")
    print(f"  python main.py --task io --io_input_path prompts/test_spanish.json \\")
    print(f"    --raw_data_dir raw_data/private/wiki_spanish \\")
    print(f"    --io_output_root eval_data/spanish_test --api hf \\")
    print(f"    --hf_ckpt meta-llama/Llama-2-7b-chat-hf --is_chat_model true")

if __name__ == "__main__":
    create_test_spanish_data()



