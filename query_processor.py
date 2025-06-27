import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
import pinecone

class QueryProcessor:
    def __init__(self, physics_index, chemistry_index):
        # Load environment variables first
        self.physics_index = physics_index
        self.chemistry_index = chemistry_index
        load_dotenv()
        
        # Configure with explicit error handling
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def classify_subject(self, query: str) -> str:
        """Determine if the query is about physics or chemistry"""
        prompt = f"""
        Analyze the following query and determine if it's about physics or chemistry.
        Return ONLY one word: either "physics" or "chemistry".
        
        Query: "{query}"
        
        Consider:
        1. Subject-specific terminology
        2. Topic areas (e.g., mechanics, electricity for physics; reactions, elements for chemistry)
        3. Context clues
        
        If the query could be about either subject or is unclear, return "physics" as default.
        """
        
        try:
            response = self.model.generate_content(prompt)
            subject = response.text.strip().lower()
            return subject if subject in ["physics", "chemistry"] else "physics"
        except Exception as e:
            print(f"Error classifying subject: {str(e)}")
            return "physics"  # Default to physics on error

    def get_appropriate_index(self, subject: str) -> Index:
        """Return the appropriate index based on subject"""
        return self.physics_index if subject == "physics" else self.chemistry_index

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
            # print(parsed_filters)
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

    def search_questions(self, query: str, top_k=10, relevance_threshold=0.5) -> list:
        """Search Pinecone with query filters and semantic search"""
        # First classify the subject
        subject = self.classify_subject(query)
        index = self.get_appropriate_index(subject)
        
        parsed = self.parse_query(query)
        filters = parsed.get("filters", {})
        
        # Generate search embedding
        search_embed = genai.embed_content(
            model="models/text-embedding-004",
            content=parsed.get("search_text", ""),
            task_type="retrieval_query"
        )["embedding"]
        
        # If filters are present, use simple top_k approach
        if filters:
            return index.query(
                vector=search_embed,
                filter=filters,
                top_k=top_k,
                include_metadata=True,
                hybrid=True,
                alpha=0.5
            )["matches"]
        
        # If no filters, use batch approach with relevance threshold
        else:
            # Start with a reasonable batch size
            batch_size = 5
            all_matches = []
            last_score = 1.0  # Start with perfect score
            
            while last_score >= relevance_threshold:
                # Query Pinecone with current batch
                results = index.query(
                    vector=search_embed,
                    filter=filters,  # No filters
                    top_k=batch_size,
                    include_metadata=True,
                    hybrid=True,
                    alpha=0.5
                )["matches"]
                
                if not results:  # No more results
                    break
                    
                # Get the last score in this batch
                last_score = results[-1].score
                
                # Add results to our collection
                all_matches.extend(results)
                
                # If we got fewer results than requested, we're done
                if len(results) < batch_size:
                    break
                    
                # Increase batch size for next iteration
                batch_size *= 2
            
            # Filter out results below threshold and limit to top_k
            filtered_matches = [match for match in all_matches if match.score >= relevance_threshold]
            return filtered_matches[:50]

# quer_proc=QueryProcessor()
# result=quer_proc.parse_query("give me question from year 2023")
# print(result)
