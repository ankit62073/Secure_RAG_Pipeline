import numpy as np
from typing import List, Dict, Tuple, Literal, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import defaultdict

QueryType = Literal["SQL", "RAG"]

@dataclass
class QueryIntent:
    intent_type: str
    confidence: float
    entities: List[Dict]
    structured_data_references: List[str]
    temporal_references: List[str]
    aggregation_references: List[str]

class AdvancedQueryClassifier:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the advanced query classifier with various NLP components
        """
        # Load models and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize semantic templates
        self.initialize_semantic_templates()
        
        # Load custom entity patterns
        self.custom_entity_patterns = self.load_custom_entity_patterns()
        
        # Initialize few-shot examples
        self.few_shot_examples = self.initialize_few_shot_examples()
        
        # Context window for historical queries
        self.query_history = []
        self.max_history = 5

    def initialize_semantic_templates(self):
        """
        Initialize semantic templates for different query types
        """
        self.sql_templates = [
            "Show me {aggregation} of {entity} grouped by {dimension}",
            "Calculate {metric} for {entity} where {condition}",
            "List all {entity} sorted by {metric} in {order}",
            "Find {entity} between {date_range}",
        ]
        
        self.rag_templates = [
            "Explain the concept of {topic}",
            "What are the main ideas in {document}",
            "Summarize the {content_type}",
            "What is the relationship between {topic1} and {topic2}",
        ]

    def load_custom_entity_patterns(self) -> Dict:
        """
        Load custom entity patterns for domain-specific recognition
        """
        return {
            'metrics': [
                'revenue', 'sales', 'profit', 'margin', 'growth',
                'conversion rate', 'churn rate', 'retention'
            ],
            'dimensions': [
                'product', 'category', 'region', 'customer', 'segment',
                'channel', 'department'
            ],
            'temporal': [
                'daily', 'weekly', 'monthly', 'quarterly', 'yearly',
                'MTD', 'YTD', 'QoQ', 'YoY'
            ]
        }

    def initialize_few_shot_examples(self) -> List[Dict]:
        """
        Initialize few-shot examples for better context
        """
        return [
            {
                'question': 'What was the total revenue by product category in Q1?',
                'type': 'SQL',
                'features': {
                    'has_aggregation': True,
                    'has_dimension': True,
                    'has_temporal': True,
                    'structured_data_reference': True
                }
            },
            {
                'question': 'Can you explain how machine learning algorithms work?',
                'type': 'RAG',
                'features': {
                    'has_aggregation': False,
                    'has_dimension': False,
                    'has_temporal': False,
                    'structured_data_reference': False
                }
            }
        ]

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using transformer model
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def extract_query_features(self, question: str) -> Dict:
        """
        Extract comprehensive features from the query
        """
        doc = self.nlp(question.lower())
        
        features = {
            'entities': [],
            'noun_chunks': [],
            'verb_phrases': [],
            'dependency_patterns': [],
            'custom_entities': defaultdict(list)
        }
        
        # Extract named entities
        features['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract noun chunks
        features['noun_chunks'] = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract verb phrases
        features['verb_phrases'] = [token.text for token in doc if token.pos_ == "VERB"]
        
        # Extract dependency patterns
        features['dependency_patterns'] = [
            (token.text, token.dep_, token.head.text)
            for token in doc
        ]
        
        # Match custom entity patterns
        for category, patterns in self.custom_entity_patterns.items():
            for pattern in patterns:
                if pattern in question.lower():
                    features['custom_entities'][category].append(pattern)
        
        return features

    def semantic_similarity_score(self, query: str, template_type: str) -> float:
        """
        Calculate semantic similarity score against templates
        """
        query_embedding = self.get_embedding(query)
        templates = (self.sql_templates if template_type == "SQL" 
                    else self.rag_templates)
        
        similarities = []
        for template in templates:
            template_embedding = self.get_embedding(template)
            similarity = cosine_similarity(query_embedding, template_embedding)[0][0]
            similarities.append(similarity)
        
        return max(similarities)

    def analyze_query_intent(self, question: str) -> QueryIntent:
        """
        Analyze the query intent using multiple approaches
        """
        features = self.extract_query_features(question)
        
        # Calculate semantic similarities
        sql_similarity = self.semantic_similarity_score(question, "SQL")
        rag_similarity = self.semantic_similarity_score(question, "RAG")
        
        # Extract structured data references
        structured_refs = [
            chunk for chunk in features['noun_chunks']
            if any(dim in chunk.lower() for dim in self.custom_entity_patterns['dimensions'])
        ]
        
        # Extract temporal references
        temporal_refs = features['custom_entities']['temporal']
        
        # Extract aggregation references
        aggregation_refs = [
            token for token in features['verb_phrases']
            if token.lower() in ['calculate', 'sum', 'average', 'count']
        ]
        
        return QueryIntent(
            intent_type="SQL" if sql_similarity > rag_similarity else "RAG",
            confidence=max(sql_similarity, rag_similarity),
            entities=features['entities'],
            structured_data_references=structured_refs,
            temporal_references=temporal_refs,
            aggregation_references=aggregation_refs
        )

    def get_contextual_features(self, question: str) -> Dict:
        """
        Extract contextual features considering query history
        """
        contextual_features = {
            'historical_pattern': None,
            'topic_continuity': False,
            'reference_resolution': []
        }
        
        if self.query_history:
            # Check for topic continuity
            current_entities = set(self.extract_query_features(question)['entities'])
            previous_entities = set(
                self.extract_query_features(self.query_history[-1])['entities']
            )
            contextual_features['topic_continuity'] = bool(
                current_entities.intersection(previous_entities)
            )
            
            # Analyze reference resolution
            doc = self.nlp(question)
            for token in doc:
                if token.pos_ == "PRON":
                    contextual_features['reference_resolution'].append(token.text)
        
        return contextual_features

    def classify_with_context(self, question: str, context: Optional[Dict] = None) -> Tuple[QueryType, float, Dict]:
        """
        Classify query with contextual awareness
        """
        # Get query intent
        intent = self.analyze_query_intent(question)
        
        # Get contextual features
        contextual_features = self.get_contextual_features(question)
        
        # Combine evidence
        evidence = {
            'intent_confidence': intent.confidence,
            'has_structured_refs': bool(intent.structured_data_references),
            'has_temporal_refs': bool(intent.temporal_references),
            'has_aggregation': bool(intent.aggregation_references),
            'context_continuation': contextual_features['topic_continuity'],
            'has_references': bool(contextual_features['reference_resolution'])
        }
        
        # Calculate final confidence score
        confidence_weights = {
            'intent_confidence': 0.4,
            'has_structured_refs': 0.2,
            'has_temporal_refs': 0.15,
            'has_aggregation': 0.15,
            'context_continuation': 0.05,
            'has_references': 0.05
        }
        
        final_confidence = sum(
            evidence[key] * confidence_weights[key]
            for key in confidence_weights
        )
        
        # Update query history
        self.query_history.append(question)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
        
        query_type: QueryType = intent.intent_type
        return query_type, final_confidence, evidence

    def explain_classification(self, question: str) -> str:
        """
        Provide detailed explanation of classification decision
        """
        query_type, confidence, evidence = self.classify_with_context(question)
        intent = self.analyze_query_intent(question)
        
        explanation = [
            f"Classification: {query_type} (confidence: {confidence:.2f})\n",
            "Evidence:",
            f"- Intent confidence: {evidence['intent_confidence']:.2f}",
            f"- Structured data references: {', '.join(intent.structured_data_references) or 'None'}",
            f"- Temporal references: {', '.join(intent.temporal_references) or 'None'}",
            f"- Aggregation references: {', '.join(intent.aggregation_references) or 'None'}",
            f"- Context continuation: {'Yes' if evidence['context_continuation'] else 'No'}",
            f"- Reference resolution needed: {'Yes' if evidence['has_references'] else 'No'}"
        ]
        
        return "\n".join(explanation)


def test_advanced_classifier():
    """
    Test the advanced classifier with various types of queries
    """
    classifier = AdvancedQueryClassifier()
    
    test_cases = [
        # SQL-like queries
        "What was the total revenue by product category in Q1 2023?",
        "Show me the daily active users trend for the past month",
        "Calculate the average order value by customer segment",
        
        # RAG-like queries
        "Explain the concept of neural networks in machine learning",
        "What are the key findings from the latest market research report?",
        "How does our product compare to competitor offerings?",
        
        # Ambiguous queries
        "What are our best performing products?",
        "Show me customer feedback for Q2",
        "How has the market changed since last year?",
        
        # Context-dependent queries
        "How does it compare to last quarter?",
        "What about the other segments?",
        "Can you break this down by region?"
    ]
    
    print("Advanced Query Classification Test Results:")
    print("-" * 60)
    
    for question in test_cases:
        print(f"\nQuestion: {question}")
        print(classifier.explain_classification(question))
        print("-" * 40)

if __name__ == "__main__":
    test_advanced_classifier()