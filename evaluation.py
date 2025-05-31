import os
from dotenv import load_dotenv
from pinecone import Pinecone
from query_processor import QueryProcessor
import google.generativeai as genai
from typing import List, Dict, Set
import numpy as np

# Load environment variables
load_dotenv()

# Configuration functions
def configure_gemini():
    """Configure and return Gemini model"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def initialize_pinecone():
    """Initialize and return Pinecone index connection"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index("o-level-physics-paper-1")

# Evaluation class
class RagEvaluator:
    def __init__(self, test_data):
        self.test_data = test_data
        self.index = initialize_pinecone()
        self.query_processor = QueryProcessor(self.index)
        self.gemini_model = configure_gemini()
        
    def _doc_id_from_meta(self, meta):
        """Construct document ID from metadata"""
        return f"{meta['year']}_Variant{meta['variant']}_Q{meta['questionNumber']}"

    def _calculate_metrics(self, retrieved: List[str], relevant: Set[str], query_type: str) -> Dict[str, float]:
        """Calculate all evaluation metrics for a single query"""
        metrics = {}
        
        # Basic metrics
        if query_type == "year":
            # For year-based queries, only check if the year matches
            year = next(iter(relevant)).split("_")[0]  # Get year from first relevant doc
            relevant_retrieved = len([doc for doc in retrieved if doc.startswith(year)])
            total_relevant = len(relevant)
            total_retrieved = len(retrieved)
        else:
            # For topic and mixed queries, use exact document matching
            relevant_retrieved = len(relevant & set(retrieved))
            total_relevant = len(relevant)
            total_retrieved = len(set(retrieved))
    
        print(f"total_retrieved: {total_retrieved}")
        print(f"relevant_retrieved: {relevant_retrieved}")
        print('\n\n\n\n')
        
        # Precision@k
        metrics['precision@10'] = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        
        # Recall@k
        metrics['recall@10'] = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        # F1-score@k
        if metrics['precision@10'] + metrics['recall@10'] > 0:
            metrics['f1@10'] = 2 * (metrics['precision@10'] * metrics['recall@10']) / (metrics['precision@10'] + metrics['recall@10'])
        else:
            metrics['f1@10'] = 0
            
        # MRR (Mean Reciprocal Rank)
        if query_type == "year":
            # For year-based queries, find first document with matching year
            year = next(iter(relevant)).split("_")[0]
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id.startswith(year):
                    metrics['mrr'] = 1 / rank
                    break
            else:
                metrics['mrr'] = 0
        else:
            # For other queries, use exact document matching
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    metrics['mrr'] = 1 / rank
                    break
            else:
                metrics['mrr'] = 0
            
        # Accuracy (exact match of all relevant documents in top k)
        # if query_type == "year":
        #     # For year-based queries, check if all retrieved docs are from the correct year
        #     year = next(iter(relevant)).split("_")[0]
        #     metrics['accuracy@10'] = 1.0 if all(doc.startswith(year) for doc in retrieved) else 0.0
        # else:
        #     metrics['accuracy@10'] = 1.0 if relevant.issubset(set(retrieved)) else 0.0
        
        return metrics

    def evaluate(self):
        """Evaluate the RAG system on all test cases"""
        all_metrics = {
            'precision@10': [],
            'recall@10': [],
            'f1@10': [],
            'mrr': [],
            # 'accuracy@10': []  # Commented out accuracy metric
        }
        
        # Track metrics by query type
        query_type_metrics = {
            'topic': {'precision@10': [], 'recall@10': [], 'f1@10': []},
            'year': {'precision@10': [], 'recall@10': [], 'f1@10': []},
            'mixed': {'precision@10': [], 'recall@10': [], 'f1@10': []}
        }
        
        # i=0
        for test_case in self.test_data:
            query = test_case['query']
            relevant = set(test_case['relevant_docs'])
            query_type = test_case['query_type']
            
            # Get search results
            results = self.query_processor.search_questions(query)
            retrieved = [self._doc_id_from_meta(match.metadata) for match in results[:20]]
            # i=i+1
            # print(f"{i}: retrieved ids: {retrieved}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(retrieved, relevant, query_type)
            
            # Store overall metrics
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
                
            # Store query type specific metrics
            for metric in ['precision@10', 'recall@10', 'f1@10']:
                query_type_metrics[query_type][metric].append(metrics[metric])
        
        # Calculate averages
        results = {
            metric: np.mean(values) for metric, values in all_metrics.items()
        }
        
        # Add query type specific results
        results['by_query_type'] = {
            qtype: {
                metric: np.mean(values) for metric, values in metrics.items()
            }
            for qtype, metrics in query_type_metrics.items()
        }
        
        return results

# Comprehensive test data covering different query types
TEST_DATA = [
    # Topic-based queries
    {
        "query": "Find questions about vectors and scalars",
        "relevant_docs": [
            "2023_Variant11_Q1",  # Which description is of a scalar quantity?
            "2022_Variant11_Q2",  # Velocity is given by the change in displacement divided by the change in time
            "2021_Variant11_Q1",  # Which quantity is a vector?
            "2021_Variant12_Q2",  # A student investigates the motion of a ball falling through the air. Which quantity is a vector?
            "2020_Variant12_Q1",  # Which quantity is a vector?
            "2020_Variant11_Q8",  # Mass and weight vectors question
            "2019_Variant11_Q1",  # Which word is the name of a scalar quantity?
            "2019_Variant12_Q1",  # Vectors and scalars question
            "2019_Variant11_Q1",  # Which quantities are both vectors?
            "2021_Variant11_Q1",  # A list of various quantities is shown. How many of these quantities are vectors?
            "2020_Variant11_Q8",  # Which statement about mass and weight is correct?
            "2021_Variant12_Q1"   # A boy starts at P and walks 3.0 m due north from P to Q and then 4.0 m due east from Q to R
        ],
        "query_type": "topic"
    },

    {
        "query": "Find questions about forces and friction",
        "relevant_docs": [
            "2023_Variant11_Q3",  # Friction on wooden block
            "2022_Variant11_Q1",  # Resultant forces
            "2022_Variant11_Q6",  # Frictional effects on cyclist
            "2022_Variant12_Q8",  # Forces in circular motion
            "2021_Variant11_Q2",  # Resultant forces
            "2021_Variant12_Q4",  # Forces in equilibrium
            "2021_Variant12_Q5",  # Forces in circular motion
            "2020_Variant11_Q3",  # Friction and driving force
            "2020_Variant11_Q6",  # Friction on slope
            "2020_Variant11_Q11", # Forces changing motion
            "2020_Variant12_Q6",  # Forces opposing motion
            "2020_Variant12_Q8",  # Forces in circular motion
            "2019_Variant11_Q2",  # Resultant forces
            "2019_Variant11_Q6",  # Balanced forces
            "2019_Variant11_Q7",  # Newton's laws and forces
            "2019_Variant12_Q7",  # Friction on slope
            "2020_Variant11_Q4",  # Car braking and friction
            "2020_Variant12_Q4"   # Car skidding and friction
        ],
        "query_type": "topic"
    },
    {
        "query": "Find questions about electricity and circuits",
        "relevant_docs": [
            "2023_Variant11_Q25",  # Circuit with lamp not lighting
            "2023_Variant11_Q26",  # Resistors in series and parallel
            "2023_Variant11_Q27",  # Electric motor power calculation
            "2023_Variant11_Q28",  # Double insulation and safety
            "2022_Variant11_Q30",  # Variable resistor in circuit
            "2022_Variant11_Q31",  # Brightness of lamps in circuits
            "2022_Variant11_Q32",  # Fuse purpose in electrical appliance
            "2022_Variant11_Q33",  # Transformer operation
            "2021_Variant11_Q29",  # Electric circuits
            "2021_Variant11_Q30",  # Electromagnetic spectrum
            "2021_Variant11_Q31",  # Electrostatics
            "2021_Variant11_Q32",  # Magnetism
            "2020_Variant11_Q35",  # Potential difference in circuit
            "2020_Variant11_Q36",  # Resistors in series and parallel
            "2020_Variant11_Q37",  # Current in parallel circuits
            "2020_Variant11_Q38",  # Potential divider with thermistor and LDR
            "2019_Variant11_Q34",  # Series and parallel resistors
            "2019_Variant11_Q35",  # Ammeter connection
            "2019_Variant11_Q36",  # Electrical safety and wiring
            "2019_Variant11_Q37"   # Transformer turns ratio
        ],
        "query_type": "topic"
    },
    
    # Year-based queries
    {
        "query": "Show me questions from 2023 paper",
        "relevant_docs": [
            "2023_Variant11_Q1", "2023_Variant11_Q2", "2023_Variant11_Q3",
            "2023_Variant11_Q4", "2023_Variant11_Q5", "2023_Variant11_Q6",
            "2023_Variant12_Q1", "2023_Variant12_Q2", "2023_Variant12_Q3",
            "2023_Variant12_Q4", "2023_Variant12_Q5", "2023_Variant12_Q6"
        ],
        "query_type": "year"
    },
    {
        "query": "Find questions from 2022 exam",
        "relevant_docs": [
            "2022_Variant11_Q1", "2022_Variant11_Q2", "2022_Variant11_Q3",
            "2022_Variant11_Q4", "2022_Variant11_Q5", "2022_Variant11_Q6",
            "2022_Variant12_Q1", "2022_Variant12_Q2", "2022_Variant12_Q3",
            "2022_Variant12_Q4", "2022_Variant12_Q5", "2022_Variant12_Q6"
        ],
        "query_type": "year"
    },
    {
        "query": "Show me questions from 2021 exam",
        "relevant_docs": [
            "2021_Variant11_Q1", "2021_Variant11_Q2", "2021_Variant11_Q3",
            "2021_Variant11_Q4", "2021_Variant11_Q5", "2021_Variant11_Q6",
            "2021_Variant12_Q1", "2021_Variant12_Q2", "2021_Variant12_Q3",
            "2021_Variant12_Q4", "2021_Variant12_Q5", "2021_Variant12_Q6"
        ],
        "query_type": "year"
    },
    {
        "query": "Find questions from 2020 exam",
        "relevant_docs": [
            "2020_Variant11_Q1", "2020_Variant11_Q2", "2020_Variant11_Q3",
            "2020_Variant11_Q4", "2020_Variant11_Q5", "2020_Variant11_Q6",
            "2020_Variant12_Q1", "2020_Variant12_Q2", "2020_Variant12_Q3",
            "2020_Variant12_Q4", "2020_Variant12_Q5", "2020_Variant12_Q6"
        ],
        "query_type": "year"
    },
    {
        "query": "Show me questions from 2019 exam",
        "relevant_docs": [
            "2019_Variant11_Q1", "2019_Variant11_Q2", "2019_Variant11_Q3",
            "2019_Variant11_Q4", "2019_Variant11_Q5", "2019_Variant11_Q6",
            "2019_Variant12_Q1", "2019_Variant12_Q2", "2019_Variant12_Q3",
            "2019_Variant12_Q4", "2019_Variant12_Q5", "2019_Variant12_Q6"
        ],
        "query_type": "year"
    },
    
    # Mixed queries (topic + year)
    {
        "query": "Find questions about radioactivity from 2022 papers",
        "relevant_docs": [
            "2022_Variant11_Q36",  # Types of radiation (alpha, gamma, beta)
            "2022_Variant11_Q37",  # Electrons in magnetic field
            "2022_Variant11_Q38",  # Radioactivity characteristics
            "2022_Variant11_Q39",  # Half-life definition
            "2022_Variant11_Q40",  # Radioactivity shielding
            "2022_Variant12_Q37",  # Types of radiation
            "2022_Variant12_Q38",  # Relay and electromagnetism
            "2022_Variant12_Q39",  # Alpha-particle characteristics
            "2022_Variant12_Q40",  # Carbon dating
            "2022_Variant11_Q38",  # Radioactive emissions characteristics
            "2022_Variant12_Q37"   # Three types of radiation
        ],
        "query_type": "mixed"
    },
    {
        "query": "Show me questions about optics and lenses from 2023",
        "relevant_docs": [
            "2023_Variant11_Q18",  # Plane mirror image
            "2023_Variant11_Q19",  # Light refraction
            "2023_Variant11_Q20",  # Total internal reflection
            "2023_Variant11_Q21",  # Converging lens
            "2023_Variant12_Q27",  # Light refraction
            "2023_Variant12_Q28",  # Total internal reflection
            "2023_Variant12_Q29",  # Lens magnification
            "2023_Variant12_Q30",  # Electromagnetic spectrum
            "2023_Variant11_Q29",  # Parallel beam through converging lens
            "2023_Variant12_Q24",  # Diverging/Converging lens image
            "2023_Variant12_Q25",  # Long-sighted vision correction
            "2023_Variant12_Q26"   # Light dispersion through prism
        ],
        "query_type": "mixed"
    },
    {
        "query": "Find questions about energy and power from 2020",
        "relevant_docs": [
            "2020_Variant11_Q14",  # Energy transfer in resistor
            "2020_Variant11_Q15",  # Conservation of energy
            "2020_Variant11_Q16",  # Renewable/non-renewable energy
            "2020_Variant11_Q39",  # Electrical energy calculation
            "2020_Variant12_Q19",  # Coal-fired power station stages
            "2020_Variant12_Q20",  # Energy transfer situations
            "2020_Variant12_Q21",  # Electric car battery charging
            "2020_Variant11_Q16",  # Energy sources
            "2020_Variant12_Q18",  # Nuclear fusion in Sun
            "2020_Variant11_Q14",  # Energy conversion in circuits
            "2020_Variant12_Q20"   # Energy transfer scenarios
        ],
        "query_type": "mixed"
    }
]

# Run evaluation
if __name__ == "__main__":
    evaluator = RagEvaluator(TEST_DATA)
    results = evaluator.evaluate()
    
    print("\nOverall Metrics:")
    print(f"Average Precision: {results['precision@10']:.3f}")
    print(f"Average Recall: {results['recall@10']:.3f}")
    print(f"Average F1: {results['f1@10']:.3f}")
    print(f"Average MRR: {results['mrr']:.3f}")
    # print(f"Average Accuracy@10: {results['accuracy@10']:.3f}")  # Commented out accuracy print
    
    print("\nMetrics by Query Type:")
    for query_type, metrics in results['by_query_type'].items():
        print(f"\n{query_type.upper()} Queries:")
        print(f"Precision: {metrics['precision@10']:.3f}")
        print(f"Recall: {metrics['recall@10']:.3f}")
        print(f"F1: {metrics['f1@10']:.3f}")
