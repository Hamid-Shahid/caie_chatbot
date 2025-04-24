import os
from dotenv import load_dotenv
from pinecone import Pinecone
from query_processor import QueryProcessor

def initialize_pinecone():
    """Initialize and return Pinecone index connection"""
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index("o-level-physics-paper-1")

def main():
    # Initialize services
    index = initialize_pinecone()
    query_processor = QueryProcessor(index)
    
    # Example search
    query = "Find questions on magnetism"
    results = query_processor.search_questions(query)
    
    # Display results
    print(f"Results for: '{query}'\n")
    for i, match in enumerate(results, 1):
        meta = match.metadata
        print(f"year: {meta['year']}")
        print(f"variant: {meta['variant']}")
        print(f"{i}. Question: {meta['questionNumber']}")
        print(f"   Statement: {meta['questionStatement']}")
        print(f"   Diagram: {meta['image']}")
        print(f"   options: {meta['options']}")
        # print(f"   Score: {match.score:.3f}\n")

if __name__ == "__main__":
    main()