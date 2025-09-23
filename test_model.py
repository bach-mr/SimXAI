import json
import yaml
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import requests
from urllib.parse import urlparse
import numpy as np
from collections import defaultdict
import hashlib
import pickle


@dataclass
class CardInfo:
    """Structure to hold extracted card information"""
    card_type: str  # "model" or "data"
    title: str
    sections: Dict[str, Any]
    metadata: Dict[str, Any]
    raw_content: str
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryResult:
    """Structure for query results"""
    query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    confidence_score: float
    sources: List[str]


class SimpleEmbedding:
    """Simple embedding system using TF-IDF like approach"""
    
    def __init__(self):
        self.vocabulary = set()
        self.idf_scores = {}
        self.documents = []
    
    def fit(self, documents: List[str]):
        """Fit the embedding model on documents"""
        self.documents = documents
        word_doc_count = defaultdict(int)
        
        # Build vocabulary and document frequency
        for doc in documents:
            words = set(self._tokenize(doc))
            self.vocabulary.update(words)
            for word in words:
                word_doc_count[word] += 1
        
        # Calculate IDF scores
        total_docs = len(documents)
        for word in self.vocabulary:
            self.idf_scores[word] = np.log(total_docs / (word_doc_count[word] + 1))
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        words = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        vocab_list = sorted(list(self.vocabulary))
        word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
        
        # Calculate TF-IDF
        word_count = defaultdict(int)
        for word in words:
            if word in self.vocabulary:
                word_count[word] += 1
        
        for word, count in word_count.items():
            if word in word_to_idx:
                tf = count / len(words) if words else 0
                tfidf = tf * self.idf_scores[word]
                vector[word_to_idx[word]] = tfidf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if len(word) > 2]
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class InteractiveCardAgent:
    """
    Enhanced LLM-based agent for interactive querying of model and data cards using RAG.
    """
    
    def __init__(self):
        self.model_card_sections = {
            'model_details', 'intended_use', 'factors', 'metrics', 'evaluation_data',
            'training_data', 'quantitative_analyses', 'ethical_considerations',
            'caveats_recommendations', 'technical_specifications', 'model_architecture',
            'training_procedure', 'performance', 'limitations', 'bias_analysis'
        }
        
        self.data_card_sections = {
            'dataset_overview', 'data_sources', 'data_collection', 'data_preprocessing',
            'data_quality', 'privacy_security', 'ethical_considerations', 'limitations',
            'usage_guidelines', 'maintenance', 'distribution', 'licensing'
        }
        
        self.loaded_cards: Dict[str, CardInfo] = {}
        self.embedder = SimpleEmbedding()
        self.is_fitted = False
        
        # Question templates for better understanding
        self.question_templates = {
            'performance': ['accuracy', 'score', 'benchmark', 'metric', 'evaluation', 'result'],
            'training': ['train', 'dataset', 'data', 'corpus', 'learning'],
            'architecture': ['model', 'architecture', 'structure', 'layer', 'parameter'],
            'limitations': ['limit', 'constraint', 'problem', 'issue', 'weakness'],
            'usage': ['use', 'application', 'purpose', 'intended', 'task'],
            'ethical': ['bias', 'fair', 'ethical', 'responsible', 'harm'],
            'technical': ['implementation', 'code', 'library', 'framework', 'requirement'],
            'license': ['license', 'copyright', 'permission', 'legal', 'terms']
        }
    
    def load_card_from_file(self, file_path: str, card_id: Optional[str] = None) -> str:
        """Load a card from a local file and return card ID"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Card file not found: {file_path}")
        
        content = path.read_text(encoding='utf-8')
        file_extension = path.suffix.lower()
        
        card_info = self._parse_card_content(content, file_extension)
        card_id = card_id or self._generate_card_id(file_path)
        
        self._process_card_for_rag(card_info)
        self.loaded_cards[card_id] = card_info
        self._refit_embedder()
        
        return card_id
    
    def load_card_from_url(self, url: str, card_id: Optional[str] = None) -> str:
        """Load a card from a URL and return card ID"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text
            
            parsed_url = urlparse(url)
            if parsed_url.path.endswith('.json'):
                file_extension = '.json'
            elif parsed_url.path.endswith('.yaml') or parsed_url.path.endswith('.yml'):
                file_extension = '.yaml'
            else:
                file_extension = '.md'
            
            card_info = self._parse_card_content(content, file_extension)
            card_id = card_id or self._generate_card_id(url)
            
            self._process_card_for_rag(card_info)
            self.loaded_cards[card_id] = card_info
            self._refit_embedder()
            
            return card_id
        
        except requests.RequestException as e:
            raise Exception(f"Failed to load card from URL: {e}")
    
    def load_huggingface_card(self, model_or_dataset_name: str, card_type: str = "model") -> str:
        """Load a card from Hugging Face Hub and return card ID"""
        if card_type not in ["model", "dataset"]:
            raise ValueError("card_type must be 'model' or 'dataset'")
        
        base_url = "https://huggingface.co"
        if card_type == "model":
            url = f"{base_url}/{model_or_dataset_name}/raw/main/README.md"
        else:
            url = f"{base_url}/datasets/{model_or_dataset_name}/raw/main/README.md"
        
        return self.load_card_from_url(url, model_or_dataset_name)
    
    def ask_question(self, question: str, card_id: Optional[str] = None, top_k: int = 3) -> QueryResult:
        """Ask a question about loaded cards using RAG"""
        if not self.loaded_cards:
            return QueryResult(
                query=question,
                answer="No cards have been loaded. Please load a model or data card first.",
                relevant_chunks=[],
                confidence_score=0.0,
                sources=[]
            )
        
        # Determine which cards to search
        cards_to_search = [card_id] if card_id and card_id in self.loaded_cards else list(self.loaded_cards.keys())
        
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(question, cards_to_search, top_k)
        
        # Generate answer
        answer = self._generate_answer(question, relevant_chunks)
        
        # Calculate confidence
        confidence = self._calculate_confidence(question, relevant_chunks)
        
        # Extract sources
        sources = list(set([chunk['source'] for chunk in relevant_chunks]))
        
        return QueryResult(
            query=question,
            answer=answer,
            relevant_chunks=relevant_chunks,
            confidence_score=confidence,
            sources=sources
        )
    
    def start_interactive_session(self):
        """Start an interactive question-answering session"""
        print("🤖 Interactive Card Agent - Ready to answer your questions!")
        print("Commands:")
        print("  'load <path_or_url>' - Load a new card")
        print("  'list' - List loaded cards")
        print("  'help' - Show help")
        print("  'quit' - Exit")
        print("  Or just ask any question about your loaded cards!\n")
        
        while True:
            try:
                user_input = input("❓ Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'list':
                    self._list_loaded_cards()
                    continue
                
                elif user_input.lower().startswith('load '):
                    path_or_url = user_input[5:].strip()
                    try:
                        card_id = self._smart_load_card(path_or_url)
                        print(f"✅ Loaded card: {card_id}")
                    except Exception as e:
                        print(f"❌ Error loading card: {e}")
                    continue
                
                # Regular question
                result = self.ask_question(user_input)
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _smart_load_card(self, path_or_url: str) -> str:
        """Smart loading that detects if input is URL or file path"""
        if path_or_url.startswith(('http://', 'https://')):
            return self.load_card_from_url(path_or_url)
        elif '/' in path_or_url and not path_or_url.startswith('./'):
            # Might be a Hugging Face model/dataset name
            try:
                return self.load_huggingface_card(path_or_url, "model")
            except:
                try:
                    return self.load_huggingface_card(path_or_url, "dataset")
                except:
                    # Treat as file path
                    return self.load_card_from_file(path_or_url)
        else:
            return self.load_card_from_file(path_or_url)
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Help - Example Questions:")
        print("• 'What is the performance of this model?'")
        print("• 'What data was used for training?'")
        print("• 'What are the limitations?'")
        print("• 'How should I use this model?'")
        print("• 'What are the ethical considerations?'")
        print("• 'What is the license?'")
        print("• 'Tell me about the architecture'")
        print()
    
    def _list_loaded_cards(self):
        """List all loaded cards"""
        if not self.loaded_cards:
            print("📝 No cards loaded yet.")
            return
        
        print(f"\n📝 Loaded Cards ({len(self.loaded_cards)}):")
        for card_id, card_info in self.loaded_cards.items():
            print(f"  • {card_id} ({card_info.card_type} card): {card_info.title}")
        print()
    
    def _display_result(self, result: QueryResult):
        """Display query result in a user-friendly format"""
        print(f"\n🤖 Answer (confidence: {result.confidence_score:.1%}):")
        print(f"   {result.answer}\n")
        
        if result.sources:
            print(f"📚 Sources: {', '.join(result.sources)}")
        
        if result.relevant_chunks and result.confidence_score > 0.3:
            print(f"📋 Relevant sections found: {len(result.relevant_chunks)}")
        
        print("-" * 60)
    
    def _process_card_for_rag(self, card_info: CardInfo):
        """Process card for RAG by creating chunks and embeddings"""
        chunks = []
        
        # Create chunks from sections
        for section_name, content in card_info.sections.items():
            if isinstance(content, str) and len(content.strip()) > 0:
                chunks.append({
                    'text': content,
                    'section': section_name,
                    'source': card_info.title,
                    'card_type': card_info.card_type,
                    'type': 'section'
                })
        
        # Create chunks from metadata
        for key, value in card_info.metadata.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                chunks.append({
                    'text': f"{key}: {value}",
                    'section': 'metadata',
                    'source': card_info.title,
                    'card_type': card_info.card_type,
                    'type': 'metadata'
                })
        
        card_info.chunks = chunks
    
    def _refit_embedder(self):
        """Refit the embedder with all loaded content"""
        all_texts = []
        for card_info in self.loaded_cards.values():
            for chunk in card_info.chunks:
                all_texts.append(chunk['text'])
        
        if all_texts:
            self.embedder.fit(all_texts)
            self.is_fitted = True
            
            # Generate embeddings for all chunks
            for card_info in self.loaded_cards.values():
                for chunk in card_info.chunks:
                    chunk['embedding'] = self.embedder.embed(chunk['text'])
    
    def _retrieve_relevant_chunks(self, question: str, card_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a question"""
        if not self.is_fitted:
            return []
        
        question_embedding = self.embedder.embed(question)
        
        # Collect all chunks from specified cards
        all_chunks = []
        for card_id in card_ids:
            if card_id in self.loaded_cards:
                all_chunks.extend(self.loaded_cards[card_id].chunks)
        
        # Calculate similarities
        chunk_scores = []
        for chunk in all_chunks:
            if 'embedding' in chunk:
                similarity = self.embedder.similarity(question_embedding, chunk['embedding'])
                chunk_scores.append((chunk, similarity))
        
        # Sort by similarity and return top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in chunk_scores[:top_k] if score > 0.1]
    
    def _generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer based on retrieved chunks"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Try rephrasing or check if the information exists in the loaded cards."
        
        # Determine question category
        question_category = self._categorize_question(question)
        
        # Aggregate information by type and section
        section_info = defaultdict(list)
        metadata_info = {}
        
        for chunk in relevant_chunks:
            if chunk['type'] == 'section':
                section_info[chunk['section']].append(chunk['text'])
            elif chunk['type'] == 'metadata':
                key_value = chunk['text'].split(': ', 1)
                if len(key_value) == 2:
                    metadata_info[key_value[0]] = key_value[1]
        
        # Generate answer based on category and available information
        return self._format_answer(question, question_category, section_info, metadata_info)
    
    def _categorize_question(self, question: str) -> str:
        """Categorize the question to provide better answers"""
        question_lower = question.lower()
        
        for category, keywords in self.question_templates.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _format_answer(self, question: str, category: str, section_info: Dict, metadata_info: Dict) -> str:
        """Format the answer based on question category and available information"""
        answer_parts = []
        
        # Handle specific question categories
        if category == 'performance' and any('performance' in section.lower() or 'metric' in section.lower() or 'evaluation' in section.lower() for section in section_info.keys()):
            answer_parts.append("Based on the performance information:")
            for section, texts in section_info.items():
                if any(keyword in section.lower() for keyword in ['performance', 'metric', 'evaluation', 'result']):
                    answer_parts.append(f"• {texts[0][:200]}...")
        
        elif category == 'limitations' and any('limitation' in section.lower() or 'caveat' in section.lower() for section in section_info.keys()):
            answer_parts.append("Here are the key limitations:")
            for section, texts in section_info.items():
                if any(keyword in section.lower() for keyword in ['limitation', 'caveat', 'risk']):
                    answer_parts.append(f"• {texts[0][:200]}...")
        
        elif category == 'usage' and any('use' in section.lower() or 'intended' in section.lower() for section in section_info.keys()):
            answer_parts.append("Regarding usage:")
            for section, texts in section_info.items():
                if any(keyword in section.lower() for keyword in ['use', 'intended', 'application']):
                    answer_parts.append(f"• {texts[0][:200]}...")
        
        else:
            # General answer formatting
            if metadata_info:
                answer_parts.append("Key information:")
                for key, value in list(metadata_info.items())[:3]:
                    answer_parts.append(f"• {key}: {value}")
            
            if section_info:
                for section, texts in list(section_info.items())[:2]:
                    section_title = section.replace('_', ' ').title()
                    answer_parts.append(f"{section_title}: {texts[0][:150]}...")
        
        if not answer_parts:
            # Fallback to first relevant chunk
            if section_info:
                first_section = list(section_info.keys())[0]
                first_text = section_info[first_section][0]
                answer_parts.append(f"From the {first_section.replace('_', ' ')}: {first_text[:200]}...")
        
        return '\n'.join(answer_parts) if answer_parts else "I found some relevant information but couldn't format a specific answer. Please try a more specific question."
    
    def _calculate_confidence(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        if not relevant_chunks:
            return 0.0
        
        # Base confidence on number and quality of chunks
        base_confidence = min(0.9, len(relevant_chunks) * 0.3)
        
        # Boost confidence if question matches common patterns
        question_lower = question.lower()
        if any(template in question_lower for templates in self.question_templates.values() for template in templates):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_card_id(self, source: str) -> str:
        """Generate a unique card ID from source"""
        return hashlib.md5(source.encode()).hexdigest()[:8]
    
    # Include all the original parsing methods
    def _parse_card_content(self, content: str, file_extension: str) -> CardInfo:
        """Parse card content based on format"""
        if file_extension == '.json':
            return self._parse_json_card(content)
        elif file_extension in ['.yaml', '.yml']:
            return self._parse_yaml_card(content)
        else:  # Markdown or plain text
            return self._parse_markdown_card(content)
    
    def _parse_json_card(self, content: str) -> CardInfo:
        """Parse JSON format card"""
        try:
            data = json.loads(content)
            card_type = self._infer_card_type(data)
            
            return CardInfo(
                card_type=card_type,
                title=data.get('name', data.get('title', 'Untitled')),
                sections=data,
                metadata=data.get('metadata', {}),
                raw_content=content
            )
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
    
    def _parse_yaml_card(self, content: str) -> CardInfo:
        """Parse YAML format card"""
        try:
            data = yaml.safe_load(content)
            card_type = self._infer_card_type(data)
            
            return CardInfo(
                card_type=card_type,
                title=data.get('name', data.get('title', 'Untitled')),
                sections=data,
                metadata=data.get('metadata', {}),
                raw_content=content
            )
        except yaml.YAMLError as e:
            raise Exception(f"Invalid YAML format: {e}")
    
    def _parse_markdown_card(self, content: str) -> CardInfo:
        """Parse Markdown format card"""
        sections = {}
        metadata = {}
        title = "Untitled"
        
        # Extract YAML frontmatter if present
        yaml_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
        if yaml_match:
            try:
                metadata = yaml.safe_load(yaml_match.group(1))
                content = yaml_match.group(2)
            except yaml.YAMLError:
                pass
        
        # Extract title from first heading or metadata
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        elif 'name' in metadata:
            title = metadata['name']
        elif 'title' in metadata:
            title = metadata['title']
        
        # Split content into sections based on headings
        sections = self._extract_markdown_sections(content)
        card_type = self._infer_card_type(sections, metadata)
        
        return CardInfo(
            card_type=card_type,
            title=title,
            sections=sections,
            metadata=metadata,
            raw_content=content
        )
    
    def _extract_markdown_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content"""
        sections = {}
        
        # Split by headers (## or ###)
        parts = re.split(r'^(#{2,3})\s+(.+)$', content, flags=re.MULTILINE)
        
        if len(parts) > 1:
            for i in range(1, len(parts), 3):
                if i + 2 < len(parts):
                    header_level = parts[i]
                    header_text = parts[i + 1].strip().lower().replace(' ', '_').replace('-', '_')
                    section_content = parts[i + 2].strip()
                    if section_content:
                        sections[header_text] = section_content
        
        return sections
    
    def _infer_card_type(self, data: Union[Dict, str], metadata: Dict = None) -> str:
        """Infer whether this is a model card or data card"""
        if metadata is None:
            metadata = {}
        
        # Check metadata first
        if 'card_type' in metadata:
            return metadata['card_type']
        
        # Check for explicit indicators
        content_str = str(data).lower()
        
        model_indicators = ['model_details', 'training_procedure', 'model_architecture', 
                          'inference', 'pytorch', 'tensorflow', 'transformers', 'model performance']
        data_indicators = ['dataset', 'data_collection', 'data_sources', 'data_preprocessing',
                          'data_quality', 'annotations', 'labeling', 'samples', 'instances']
        
        model_score = sum(1 for indicator in model_indicators if indicator in content_str)
        data_score = sum(1 for indicator in data_indicators if indicator in content_str)
        
        return "model" if model_score >= data_score else "data"


# Example usage and demonstration
def main():
    """Demonstrate the interactive agent capabilities"""
    agent = InteractiveCardAgent()
    
    # Create sample cards for demonstration
    sample_model_card = """---
name: "GPT-Style Language Model"
license: "MIT"
library_name: "transformers"
model_type: "gpt2"
---

# GPT-Style Language Model

## Model Details

This is a transformer-based autoregressive language model with 117M parameters, based on the GPT-2 architecture.

## Intended Use

- Text completion and generation
- Creative writing assistance
- Code completion
- Conversational AI applications

## Training Data

The model was trained on a diverse corpus including:
- Web text from Common Crawl (40GB)
- Books corpus (8GB)
- Wikipedia articles (4GB)
- Code repositories from GitHub (2GB)

## Performance

- Perplexity on validation set: 15.2
- BLEU score on text completion: 0.78
- Human evaluation score: 7.2/10

## Limitations

- May generate biased or inappropriate content
- Performance degrades on highly technical domains
- Cannot access real-time information
- May produce factually incorrect information

## Ethical Considerations

Users should be aware that this model may perpetuate biases present in training data. Recommended to use content filtering and human oversight for production applications.
"""
    
    print("🚀 Interactive Card Agent Demo")
    print("=" * 50)
    
    # Load the sample card
    with open('temp_model_card.md', 'w') as f:
        f.write(sample_model_card)
    
    try:
        card_id = agent.load_card_from_file('temp_model_card.md')
        print(f"✅ Loaded sample card: {card_id}\n")
        
        # Demo some questions
        demo_questions = [
            "What is the performance of this model?",
            "What are the limitations?",
            "What data was used for training?",
            "How should I use this model?",
            "What are the ethical considerations?"
        ]
        
        print("📋 Demo Questions & Answers:")
        print("-" * 30)
        
        for question in demo_questions:
            result = agent.ask_question(question)
            print(f"\n❓ Q: {question}")
            print(f"🤖 A: {result.answer}")
            print(f"   (Confidence: {result.confidence_score:.1%})")
        
        print(f"\n" + "=" * 50)
        print("💬 Ready for interactive session! Try running:")
        print("   agent.start_interactive_session()")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        # Cleanup
        import os
        if os.path.exists('temp_model_card.md'):
            os.remove('temp_model_card.md')


if __name__ == "__main__":
    main()