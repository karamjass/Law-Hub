from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import pickle
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import json
from deep_translator import GoogleTranslator
import requests

app = Flask(__name__)
CORS(app)

# Cache files for different components
DATASETS_CACHE = 'legal_datasets_cache.pkl'
MATRIX_CACHE = 'search_matrix_cache.pkl'
VECTORIZER_CACHE = 'vectorizer_cache.pkl'

def setup_kaggle():
    """Setup Kaggle API credentials"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    source_file = 'D:/Users/Japneet Singh/Desktop/LawHub-1/kaggle (3).json'
    target_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(source_file):
        import shutil
        shutil.copy2(source_file, target_file)
        os.chmod(target_file, 0o600)
        print("✅ Kaggle credentials configured")
    else:
        print("⚠️  kaggle.json file not found")

def load_kaggle_datasets():
    """Load Kaggle datasets from local files"""
    datasets = {}
    
    # Updated paths to match user's actual downloaded files on D drive
    kaggle_files = {
        "us_supreme_court": "D:/Users/Japneet Singh/Desktop/database.csv",
        "bangladesh": "D:/Users/Japneet Singh/Desktop/archive (4)/bd_laws_translated_05022022.csv",
        "pakistan": "D:/Users/Japneet Singh/Desktop/archive (2)/pdf_data.json", 
        "australia": "D:/Users/Japneet Singh/Desktop/archive (3)/<main_file>.csv",  # Will be auto-detected
        "vietnamese": "D:/Users/Japneet Singh/Desktop/archive (5)/sent_truncated_dvc_train.csv",
        "legal_qa_train": "D:/Users/Japneet Singh/Desktop/archive (8)/LegalQA_Train_Dataset.csv",
        "legal_qa_test": "D:/Users/Japneet Singh/Desktop/archive (8)/LegalQA_Test_Dataset.csv"
    }
    
    # Load PDF documents from DomainDocuments
    pdf_folder = "D:/Users/Japneet Singh/Desktop/archive (8)/DomainDocuments"
    if os.path.exists(pdf_folder):
        try:
            pdf_data = []
            for file in os.listdir(pdf_folder):
                if file.endswith('.pdf'):
                    file_path = os.path.join(pdf_folder, file)
                    # For now, we'll create a simple entry for each PDF
                    # In a full implementation, you'd extract text from PDFs
                    pdf_data.append({
                        'filename': file,
                        'title': file.replace('.pdf', '').replace('-', ' '),
                        'type': 'narcotics_act',
                        'content': f"PDF Document: {file} - Narcotics Act No. 1 of 2008",
                        'file_path': file_path
                    })
            
            if pdf_data:
                datasets["narcotics_act"] = pd.DataFrame(pdf_data)
                print(f"✅ Loaded narcotics_act: {len(datasets['narcotics_act'])} PDF documents")
        except Exception as e:
            print(f"❌ Error loading PDF documents: {e}")
    
    for key, path in kaggle_files.items():
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    datasets[key] = pd.read_csv(path)
                elif path.endswith('.json'):
                    datasets[key] = pd.read_json(path)
                print(f"✅ Loaded {key}: {len(datasets[key])} rows")
            except Exception as e:
                print(f"❌ Error loading {key}: {e}")
        else:
            print(f"⚠️  File not found: {path}")
            # Try to find files in the archive folders
            if "archive" in path:
                folder_path = path.split('/')[0:-1]  # Get folder path
                folder_path = '/'.join(folder_path)
                if os.path.exists(folder_path):
                    print(f"🔍 Searching in folder: {folder_path}")
                    try:
                        for file in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, file)
                            if file.endswith('.csv') or file.endswith('.json'):
                                print(f"📁 Found file: {file}")
                                if file.endswith('.csv'):
                                    datasets[key] = pd.read_csv(file_path)
                                elif file.endswith('.json'):
                                    datasets[key] = pd.read_json(file_path)
                                print(f"✅ Loaded {key} from {file}: {len(datasets[key])} rows")
                                break
                    except Exception as e:
                        print(f"❌ Error searching folder {folder_path}: {e}")
    
    return datasets

def load_huggingface_datasets():
    """Load HuggingFace datasets"""
    datasets = {}
    
    hf_datasets = {
        "france": "louisbrulenaudet/legalkit",
        "thailand": "airesearch/WangchanX-Legal-ThaiCCL-RAG", 
        "canada": "refugee-law-lab/canadian-legal-data",
        "india": "viber1/indian-law-dataset"
    }
    
    for key, hf_name in hf_datasets.items():
        try:
            dataset = load_dataset(hf_name, split='train')
            datasets[key] = dataset.to_pandas()
            print(f"✅ Loaded {key} from HuggingFace: {len(datasets[key])} rows")
        except Exception as e:
            print(f"❌ Error loading {key} from HuggingFace: {e}")
    
    return datasets

def load_all_datasets():
    """Load all datasets with caching"""
    # Try to load from cache first
    if os.path.exists(DATASETS_CACHE):
        try:
            with open(DATASETS_CACHE, 'rb') as f:
                cached_datasets = pickle.load(f)
            print("✅ Loaded datasets from cache")
            return cached_datasets
        except:
            print("⚠️  Cache corrupted, rebuilding...")
    
    print("🔄 Loading all datasets...")
    
    # Load Kaggle datasets
    kaggle_datasets = load_kaggle_datasets()
    
    # Load HuggingFace datasets
    hf_datasets = load_huggingface_datasets()
    
    # Combine all datasets
    all_datasets = {**kaggle_datasets, **hf_datasets}
    
    # Save to cache
    try:
        with open(DATASETS_CACHE, 'wb') as f:
            pickle.dump(all_datasets, f)
        print("✅ Saved datasets to cache")
    except Exception as e:
        print(f"⚠️  Could not save datasets cache: {e}")
    
    return all_datasets

def preprocess_text_fast(text):
    """Fast text preprocessing"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    
    return text[:2000]  # Limit length for performance

def build_search_matrix(datasets):
    """Build and cache the search matrix"""
    # Check if matrix already exists
    if (os.path.exists(MATRIX_CACHE) and 
        os.path.exists(VECTORIZER_CACHE)):
        try:
            with open(MATRIX_CACHE, 'rb') as f:
                tfidf_matrix = pickle.load(f)
            with open(VECTORIZER_CACHE, 'rb') as f:
                vectorizer = pickle.load(f)
            print("✅ Loaded search matrix from cache")
            return vectorizer, tfidf_matrix
        except:
            print("⚠️  Matrix cache corrupted, rebuilding...")
    
    print("🔍 Building search matrix...")
    
    all_texts = []
    doc_index = []
    
    # Process each dataset
    for name, df in tqdm(datasets.items(), desc="Processing datasets"):
        if df.empty:
            continue
        
        print(f"  Processing {name}: {len(df)} rows")
        
        # Clean the dataframe
        df = df.fillna("")
        
        # Combine text columns intelligently
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains meaningful text
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Only use columns with substantial text
                    text_columns.append(col)
        
        if text_columns:
            # Combine meaningful text columns
            df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
        else:
            # Fallback: combine all columns
            df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
        
        # Translate non-English datasets
        if name.lower() in ["thailand", "france", "vietnamese", "china", "germany"]:
            print(f"🌍 Translating {name} dataset to English...")
            translated_rows = []
            for i, text in enumerate(df['combined_text']):
                if i % 100 == 0:  # Progress indicator
                    print(f"    Translated {i}/{len(df)} rows")
                try:
                    # Limit text length to avoid translation API limits
                    text_to_translate = str(text)[:500]
                    translated_text = GoogleTranslator(source='auto', target='en').translate(text_to_translate)
                    translated_rows.append(translated_text)
                except Exception as e:
                    print(f"    Translation error for row {i}: {e}")
                    translated_rows.append(str(text))  # Fallback to original text
            df['combined_text'] = translated_rows
            print(f"✅ Completed translation for {name}")
        
        # Preprocess and add to index
        for i, row in df.iterrows():
            text = preprocess_text_fast(row['combined_text'])
            if len(text.strip()) > 10:  # Only add non-empty texts
                all_texts.append(text)
                doc_index.append((name, i))
    
    print(f"✅ Indexed {len(all_texts)} documents")
    
    # Create TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,  # Increased for better accuracy
        ngram_range=(1, 3),  # Use trigrams for better matching
        min_df=2,  # Ignore very rare terms
        max_df=0.95,  # Ignore very common terms
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Build the matrix
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Save to cache
    try:
        with open(MATRIX_CACHE, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(VECTORIZER_CACHE, 'wb') as f:
            pickle.dump(vectorizer, f)
        print("✅ Saved search matrix to cache")
    except Exception as e:
        print(f"⚠️  Could not save matrix cache: {e}")
    
    return vectorizer, tfidf_matrix

def search_legal_documents(query, top_k=5):
    """Search through all legal documents"""
    if not hasattr(search_legal_documents, 'vectorizer') or not hasattr(search_legal_documents, 'tfidf_matrix'):
        return []
    
    # Preprocess query
    query = preprocess_text_fast(query)
    if len(query.strip()) < 3:
        return []
    
    # Vectorize query
    query_vec = search_legal_documents.vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, search_legal_documents.tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:  # Only return relevant results
            dataset_name, row_index = search_legal_documents.doc_index[idx]
            result_row = search_legal_documents.datasets[dataset_name].iloc[row_index]
            
            # Create clean result format
            result_dict = {}
            for col in result_row.index:
                value = result_row[col]
                if pd.notna(value) and str(value).strip() and len(str(value)) < 1000:
                    result_dict[col] = str(value)
            
            if result_dict:
                results.append({
                    "dataset": dataset_name.title(),
                    "score": round(similarities[idx], 4),
                    "result": result_dict
                })
    
    return results

# Initialize the system
print("🚀 Starting LawHub Backend...")

# Load datasets
datasets = load_all_datasets()
print(f"📊 Loaded {len(datasets)} datasets")

# Build search matrix
vectorizer, tfidf_matrix = build_search_matrix(datasets)

# Store in function for access
search_legal_documents.vectorizer = vectorizer
search_legal_documents.tfidf_matrix = tfidf_matrix
search_legal_documents.datasets = datasets

# Rebuild doc_index for search
all_texts = []
doc_index = []
for name, df in datasets.items():
    if df.empty:
        continue
    df = df.fillna("")
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 20:
                text_columns.append(col)
    if text_columns:
        df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    else:
        df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
    
    for i, row in df.iterrows():
        text = preprocess_text_fast(row['combined_text'])
        if len(text.strip()) > 10:
            all_texts.append(text)
            doc_index.append((name, i))

search_legal_documents.doc_index = doc_index

print(f"✅ Ready! Indexed {len(all_texts)} documents")

@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint for legal questions"""
    data = request.get_json()
    user_query = data.get('question', '')
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400
    
    results = search_legal_documents(user_query)
    return jsonify(results)

@app.route('/api/status', methods=['GET'])
def status():
    """Check backend status"""
    return jsonify({
        "status": "running",
        "datasets_loaded": len(datasets),
        "total_documents": len(all_texts),
        "datasets": list(datasets.keys()),
        "matrix_cached": os.path.exists(MATRIX_CACHE),
        "datasets_cached": os.path.exists(DATASETS_CACHE),
        "legal_qa_available": "legal_qa_train" in datasets and "legal_qa_test" in datasets,
        "narcotics_act_available": "narcotics_act" in datasets,
        "deepseek_available": bool(DEEPSEEK_API_KEY)
    })

@app.route('/api/legal_qa', methods=['POST'])
def legal_qa():
    """Specific endpoint for Legal QA questions"""
    data = request.get_json()
    user_query = data.get('question', '')
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400
    
    # Search specifically in Legal QA datasets
    results = []
    
    if "legal_qa_train" in datasets:
        train_results = search_legal_documents(user_query, top_k=3)
        for result in train_results:
            if result["dataset"] in ["Legal_qa_train", "Legal_qa_test"]:
                results.append(result)
    
    if not results:
        # Fallback to general search
        results = search_legal_documents(user_query)
    
    return jsonify(results)

@app.route('/')
def home():
    """Landing page"""
    return render_template('landing_page.html')

@app.route('/app')
def app_main():
    """Main application page"""
    return render_template('integrated_frontend.html')

# DeepSeek AI Configuration
try:
    from config import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')  # Fallback to environment variable

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def get_deepseek_legal_advice(question, context=""):
    """
    Get legal advice from DeepSeek AI
    """
    if not DEEPSEEK_API_KEY:
        return {
            "success": False,
            "error": "DeepSeek API key not configured. Please set DEEPSEEK_API_KEY environment variable."
        }
    
    # Create a comprehensive legal prompt
    legal_prompt = f"""You are an expert legal advisor with deep knowledge of international law, legal rights, and legal procedures. 

CONTEXT: {context}

USER QUESTION: {question}

Please provide comprehensive legal advice including:
1. Relevant legal rights and protections
2. Applicable laws, acts, and sections
3. Recommended actions and procedures
4. Potential legal consequences
5. When to seek professional legal counsel

Provide your response in a clear, structured format with specific legal references where applicable. Focus on practical, actionable advice while noting that this is for informational purposes and not a substitute for professional legal counsel.

RESPONSE:"""

    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert legal advisor specializing in international law, legal rights, and legal procedures. Provide comprehensive, accurate, and practical legal guidance."
                },
                {
                    "role": "user",
                    "content": legal_prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        # Try multiple connection methods to handle SSL issues
        session = requests.Session()
        
        # Method 1: Try with SSL verification disabled (for SSL issues)
        try:
            response = session.post(
                DEEPSEEK_API_URL, 
                headers=headers, 
                json=data, 
                timeout=30,
                verify=False  # Disable SSL verification
            )
        except:
            # Method 2: Try with different SSL configuration
            import ssl
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = session.post(
                DEEPSEEK_API_URL, 
                headers=headers, 
                json=data, 
                timeout=30,
                verify=False
            )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                answer = result['choices'][0]['message']['content']
                return {
                    "success": True,
                    "answer": answer,
                    "source": "DeepSeek AI Legal Advisor",
                    "model": "deepseek-chat"
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid response format from DeepSeek API"
                }
        else:
            return {
                "success": False,
                "error": f"DeepSeek API error: {response.status_code} - {response.text}"
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout. Please try again."
        }
    except requests.exceptions.SSLError as e:
        return {
            "success": False,
            "error": f"SSL Connection Error: {str(e)}. Please check your internet connection or try using a VPN."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}. Please check your internet connection."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

@app.route('/api/deepseek_legal', methods=['POST'])
def deepseek_legal():
    """Get legal advice from DeepSeek AI"""
    data = request.get_json()
    user_question = data.get('question', '')
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    # Get context from existing legal databases if available
    context = ""
    try:
        # Search existing databases for relevant context
        search_results = search_legal_documents(user_question, top_k=3)
        if search_results:
            context = "Relevant legal information from our database:\n"
            for i, result in enumerate(search_results[:2], 1):
                context += f"{i}. {result['dataset']}: {str(result['result'])[:200]}...\n"
    except:
        pass  # Continue without context if search fails
    
    # Get DeepSeek legal advice
    result = get_deepseek_legal_advice(user_question, context)
    
    # If DeepSeek fails due to insufficient balance, fallback to existing search
    if not result.get('success') and 'insufficient' in result.get('error', '').lower():
        # Fallback to existing legal database search
        fallback_results = search_legal_documents(user_question, top_k=5)
        if fallback_results:
            return jsonify({
                "success": True,
                "answer": f"⚠️ DeepSeek AI is currently unavailable (insufficient balance). Here are relevant results from our legal database:\n\n" + 
                         "\n\n".join([f"**{res['dataset']}** (Score: {res['score']}):\n{str(res['result'])[:300]}..." for res in fallback_results[:3]]),
                "source": "LawHub Legal Database (DeepSeek Fallback)",
                "fallback": True
            })
        else:
            return jsonify({
                "success": False,
                "error": "DeepSeek AI unavailable and no relevant legal information found in database. Please try the 'Legal QA' button instead."
            })
    
    return jsonify(result)

if __name__ == '__main__':
    print("🌐 Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 