import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
import json
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "o-level-chemistry-paper-1"

# Create Pinecone index (updated for v6+)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Match your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        # metadata_config removed in v6+
    )

index = pc.Index(index_name)

def process_questions(data):
    vectors = []
    for q in data["questions"]:
        # Generate embedding
        unique_id = f"{data['subjectCode']}_{data['variant']}_{data['year']}_q{q['questionNumber']}"
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            # content=q['statement'],
            content = f"""
            Exam: {data['exam']}
            Paper: {data['paper']}
            Year: {data['year']}
            Question: {q['statement']}
            Topics: {', '.join(q['topics'])}
            """,
            task_type="retrieval_document"
        )['embedding']
        months_part, year = data['year'].split()
        months = months_part.split('/')
        metadata = {
            "exam": data["exam"],
            "subjectCode": data["subjectCode"],
            "variant": data["variant"],
            "year": year,
            "months":months,
            "subject": data["subject"],
            "paper": data["paper"],
            "questionNumber": q["questionNumber"],
            "questionStatement": q["statement"],
            "options": q["options"],
            "topics": q["topics"],
            "image": q["image"]
        }
        
        vectors.append((unique_id, embedding, metadata))
    
    # Batch upsert
    for batch in [vectors[i:i+100] for i in range(0, len(vectors), 100)]:
        index.upsert(batch)

# Load data

def load_json_with_encoding(filepath):
    encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1"]
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filepath} with any supported encoding")


# directory = r"F:/FYP/Current/o-level-physics-5054-20241117T145438Z-001/jsonFormat/chem_json_format"
# # Get all files in the directory and filter JSON files
# file_paths = [
#     os.path.join(directory, file)
#     for file in os.listdir(directory)
#     if file.endswith(".json")
# ]
# for file_path in file_paths:
#     exam_data = load_json_with_encoding(file_path)
#     print("inserting data into pinecone")
#     process_questions(exam_data)
# print("Done")