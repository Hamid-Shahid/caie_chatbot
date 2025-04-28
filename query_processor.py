import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
from pinecone import Index

class QueryProcessor:
    def __init__(self,index:Index):
        # Load environment variables first
        self.index = index
        load_dotenv()  # Add this line
        
        # Configure with explicit error handling
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
    def parse_query(self, query: str) -> dict:
        """Extract filters and search text using Gemini"""
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        
        prompt = f"""
        Analyze the query and STRICTLY extract ONLY EXPLICITLY MENTIONED filters:
        - questionNumber: extract as string if specifically numbered,
        - variant: extract as string if version/variant specified,
        - subjectCode: 4-digit code as string (e.g., "5054"),
        - year,
        - months
        
        If a filter is not explicitly mentioned, OMIT IT COMPLETELY from JSON.
        
        Query: "{query}"
        
        Return JSON with:
        {{
            "filters": {{
                // ONLY include filters present in query //
                "questionNumber": "number or omit",
                "variant": "version or omit",
                "subjectCode": "4-digit code or omit",
                "year":"it could be year like 2019,2020,2021 etc if user provide only last 2 digits or say recent year then you must convert that query to full year like 19 is equals to 2019 and recent means present year - 2",
                "months": month provided in query like June, November etc if user provide short form of month like Nov Cconvert it into full form like November. if user provide multiple months take only one
            }},
            "search_text": "full original query"
        }}
        
        Examples:
        Query: "Physics paper 1 variant 12 questions"
        {{
            "filters": {{
                "variant": "12"
            }},
            "search_text": "Physics paper 1 variant 12 questions"
        }}
        
        Query: "Find question 5 from 5054"
        {{
            "filters": {{
                "questionNumber": "5",
                "subjectCode": "5054"
            }},
            "search_text": "Find question 5 from 5054"
        }}
        """
        
        try:
            response = model.generate_content(prompt)
            # print("Raw response:", response.text)
            
            # Extract JSON from markdown code block
            json_str = response.text.split('```json')[1].split('```')[0].strip()
            
            parsed_filters = json.loads(json_str)
            print(parsed_filters)
            # Format filters for Pinecone compatibility
            return {
                "filters": {
                    **{k: {"$eq": v} for k, v in parsed_filters.get("filters", {}).items() if v is not None}
                },
                "search_text": parsed_filters.get("search_text", "")
            }
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            return {"filters": {}, "search_text": query}



    def search_questions(self, query: str, top_k=10) -> list:
        """Search Pinecone with query filters and semantic search"""
        parsed = self.parse_query(query)
        # print(parsed)
        
        # Generate search embedding
        search_embed = genai.embed_content(
            model="models/text-embedding-004",
            content=parsed.get("search_text", ""),
            task_type="retrieval_query"
        )["embedding"]
        # print(50*"*")
        # print(search_embed)
        # print(50*"*")
        # Query Pinecone
        return self.index.query(
            vector=search_embed,
            filter=parsed.get("filters", {}),
            top_k=top_k,
            include_metadata=True,
            hybrid=True,  
            alpha=0.5     
        )["matches"]



# quer_proc=QueryProcessor()
# result=quer_proc.parse_query("give me question from year 2023")
# print(result)