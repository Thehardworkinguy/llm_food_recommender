# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pandas as pd
# import json
# from huggingface_hub import InferenceClient
# from sentence_transformers import SentenceTransformer
# import chromadb
# from transformers import pipeline # <-- NEW IMPORT

# # NOTE: For portability, it is highly recommended to use relative paths for your files.
# # For example, if your files are in a 'data' folder next to your script, use:
# # MODEL_PATH = "data/indian_food_classifier.pth"
# # LABELS_DIR = "data/train"
# # NUTRITION_CSV_PATH = 'data/Indian_Food_Nutrition_Processed.csv'

# # Paths (CHANGE these to your actual file locations)
# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
# NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# # Set the device for model inference
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @st.cache_resource
# def load_model_and_labels():
#     """Loads the pre-trained model and the food labels."""
#     if not os.path.exists(LABELS_DIR):
#         st.error(f"Labels folder not found at {LABELS_DIR}. Please check the path.")
#         st.stop()
        
#     labels = [d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))]
#     labels = sorted(labels)
#     if not labels:
#         st.error("No label folders found in the train directory. Model cannot be loaded.")
#         st.stop()

#     model = models.resnet18(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
#         st.stop()
        
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, labels

# try:
#     model, labels = load_model_and_labels()
# except Exception as e:
#     st.error(f"An error occurred during model loading: {e}")
#     st.stop()

# @st.cache_data
# def load_nutrition_data():
#     """Loads and preprocesses the nutrition data from a CSV file."""
#     if not os.path.exists(NUTRITION_CSV_PATH):
#         st.warning(f"Nutrition data file not found at {NUTRITION_CSV_PATH}. Nutrition lookup will not be available.")
#         return None
#     try:
#         df = pd.read_csv(NUTRITION_CSV_PATH)
#         df['Dish Name'] = df['Dish Name'].astype(str)
#         return df
#     except Exception as e:
#         st.error(f"Failed to load nutrition data from CSV: {e}")
#         return None

# nutrition_df = load_nutrition_data()

# @st.cache_resource
# def get_embedding_model_and_db():
#     """Initializes the Sentence-Transformer and ChromaDB client."""
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     client = chromadb.Client()
    
#     try:
#         collection = client.get_collection(name="nutrition_data")
#     except:
#         collection = client.create_collection(name="nutrition_data")
    
#     if collection.count() == 0 and nutrition_df is not None:
#         st.info("Ingesting data into the vector database...")
#         documents = nutrition_df['Dish Name'].tolist()
#         metadata = nutrition_df.to_dict('records')
#         ids = [f"id_{i}" for i in range(len(documents))]
        
#         collection.add(documents=documents, metadatas=metadata, ids=ids)
#         st.success("Data ingestion complete!")
        
#     return embedding_model, collection

# embedding_model, vector_db_collection = get_embedding_model_and_db()

# IMG_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# def predict(image: Image.Image):
#     """Predicts the food label from an image."""
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image)
#         return labels[output.argmax(1).item()]

# # ------------------- NEW: NER Pipeline Functions -------------------
# @st.cache_resource
# def get_ner_pipeline():
#     """Loads the NER pipeline model once and caches it."""
#     st.info("Loading Clinical NER model...")
#     ner_pipe = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
#     st.success("Clinical NER model loaded.")
#     return ner_pipe

# def run_ner_on_text(text, ner_pipeline):
#     """Runs NER on text and returns a formatted string of entities."""
#     if not text:
#         return "No clinical text provided.", []
    
#     entities = ner_pipeline(text)
#     if not entities:
#         return "No medical entities were identified.", []

#     # Format entities for the LLM prompt and for display
#     formatted_entities_list = []
#     for entity in entities:
#         entity_type = entity['entity_group']
#         entity_word = entity['word']
#         formatted_entities_list.append(f"- {entity_word} (Type: {entity_type})")
        
#     return "\n".join(formatted_entities_list), entities

# # ------------------- RAG Pipeline Functions -------------------
# def get_rag_context(food_label):
#     """RAG Step 1: Retrieves nutritional context."""
#     if vector_db_collection.count() == 0:
#         return None, None
#     results = vector_db_collection.query(query_texts=[food_label.replace('_', ' ')], n_results=1)
#     if results['metadatas'] and results['metadatas'][0]:
#         retrieved_data = results['metadatas'][0][0]
#         return retrieved_data, retrieved_data.get('Dish Name', 'N/A')
#     return None, None

# def construct_final_prompt(clinical_data, food_name, nutrition_info):
#     """RAG Step 2: Augments prompt for dietary advice."""
    
#     # Robustly handle conditions, which might be a list of strings or dicts
#     conditions_list = [str(item) for item in clinical_data.get('conditions', [])]
#     conditions = ", ".join(conditions_list) if conditions_list else "no known conditions"
    
#     # Robustly handle symptoms
#     symptoms_list = [str(item) for item in clinical_data.get('symptoms', [])]
#     symptoms = ", ".join(symptoms_list) if symptoms_list else "none"

#     # Robustly handle medications
#     medications_list = [str(item) for item in clinical_data.get('medications', [])]
#     medications = ", ".join(medications_list) if medications_list else "none"
    
#     nut_keys_to_display = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 'Free Sugar (g)',
#                            'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
#     nut_desc = ', '.join(f"{k}: {nutrition_info.get(k, 'N/A')}" for k in nut_keys_to_display)

#     return f"""You are a professional dietitian with expertise in personalized nutrition.
# I have the following data for a patient:
# - Conditions: {conditions}
# - Symptoms: {symptoms}
# - Medications: {medications}
# - The patient plans to consume: {food_name}
# - Nutritional info (per 100g): {nut_desc}

# Provide **personalized dietary advice in 4–5 sentences only**. 
# Keep it concise, patient-specific, and practical. 
# Clearly mention risks and if needed, suggest one alternative food option and also how much portion in grams or pieces can be consumed."""
# def query_hf_chat(prompt, hf_token):
#     """RAG Step 3: Generates dietary advice using Llama 3."""
#     try:
#         client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token, timeout=60)
#         response = client.chat_completion(
#             messages=[
#                 {"role": "system", "content": "You are a professional dietitian."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=400, temperature=0.7
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error during advice generation: {str(e)}"

# def extract_clinical_data(ehr_text, formatted_entities, hf_token):
#     """Extracts and structures clinical data using Llama 3, guided by NER results."""
#     if not ehr_text:
#         return {"conditions": [], "symptoms": [], "medications": []}

#     # New prompt that leverages the NER output
#     prompt_text = f"""You are a highly accurate clinical data structuring bot.
# A specialized NER model has analyzed a patient record and extracted the following entities:
# ---
# PRE-IDENTIFIED ENTITIES:
# {formatted_entities}
# ---
# ORIGINAL PATIENT RECORD:
# {ehr_text}
# ---
# Based on BOTH the pre-identified entities and the original context, your task is to accurately populate a JSON object with the keys "conditions", "symptoms", and "medications".
# Follow these rules strictly:
# 1. Use the pre-identified entities as your primary source of information.
# 2. Refer to the original text for context (e.g., to distinguish between a current symptom and a past condition).
# 3. Return ONLY the JSON object and nothing else.

# JSON Output:"""
    
#     try:
#         client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token, timeout=60)
#         response = client.chat_completion(
#             messages=[
#                 {"role": "system", "content": "You are an expert at structuring clinical data into JSON format."},
#                 {"role": "user", "content": prompt_text}
#             ],
#             max_tokens=300, temperature=0.1
#         )
#         raw_text = response.choices[0].message.content.strip()
#         print(f"DEBUG: Raw LLM output for clinical data structuring:\n{raw_text}")
        
#         try:
#             if raw_text.startswith("```json"):
#                 raw_text = raw_text.replace("```json", "").replace("```", "").strip()
#             return json.loads(raw_text)
#         except json.JSONDecodeError:
#             st.warning(f"Failed to parse JSON from LLM. Raw output: {raw_text}")
#             return {"conditions": [], "symptoms": [], "medications": [], "raw": raw_text}
#     except Exception as e:
#         st.error(f"Error during clinical data structuring: {type(e).__name__} - {str(e)}")
#         return {"conditions": [], "symptoms": [], "medications": [], "error": str(e)}

# # ------------------- Streamlit UI -------------------
# st.title("Indian Food Classifier + Personalized Diet Advice (NER + RAG)")
# st.markdown("This app uses a 2-step process: A **Clinical NER model** finds medical terms, and an **LLM** structures them and provides dietary advice.")

# # Load the NER pipeline at the start
# ner_pipeline = get_ner_pipeline()

# uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])
# ehr_text = st.text_area("Paste patient's EHR summary here (optional)", "Patient complains of a persistent headache and nausea for 2 days. History of hypertension. Prescribed Ibuprofen 400mg.", key="ehr_input")
# hf_token = st.text_input("Enter your Hugging Face API Token", type="password", key="hf_token_input")

# if uploaded_file:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     if st.button("Classify and Get Dietary Advice"):
#         if not hf_token:
#             st.warning("Please enter your Hugging Face API Token.")
#             st.stop()

#         with st.spinner("Classifying food..."):
#             pred_label = predict(img)
#             st.success(f"Predicted Food: **{pred_label.replace('_', ' ').title()}**")

#         with st.spinner("Retrieving nutritional context..."):
#             nutrition, matched_name = get_rag_context(pred_label)
        
#         if not nutrition:
#             st.warning("No nutrition data found for the predicted food. Cannot generate advice.")
#             st.stop()
        
#         st.info(f"Nutrition info matched: **{matched_name.title()}**")
        
#         st.markdown("---")
#         st.subheader("Nutrition facts (per 100g)")
#         nut_keys = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 'Free Sugar (g)',
#                       'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
#         nut_data = {k: [nutrition.get(k, 'N/A')] for k in nut_keys}
#         st.dataframe(pd.DataFrame(nut_data).T.rename(columns={0: "Amount"}), use_container_width=True)
#         st.markdown("---")

#         with st.spinner("Extracting clinical data with NER and LLM..."):
#             # Step 1: Run NER to find entities
#             ner_results_str, ner_entities = run_ner_on_text(ehr_text, ner_pipeline)
            
#             # Step 2: Use LLM to structure the data, guided by NER results
#             clinical_data = extract_clinical_data(ehr_text, ner_results_str, hf_token)
            
#             with st.expander("Show Extracted Clinical Data (2-Step Process)"):
#                 st.subheader("1. Entities Identified by Clinical NER Model")
#                 st.text(ner_results_str)
#                 st.subheader("2. Final Structured Data from Llama 3")
#                 st.json(clinical_data)
            
#             # Augment the prompt for dietary advice
#             final_prompt = construct_final_prompt(clinical_data, matched_name.title(), nutrition)
            
#             with st.expander("Show Final Prompt for Diet Advice"):
#                 st.code(final_prompt)

#             # Generate the final dietary advice
#             advice = query_hf_chat(final_prompt, hf_token)
#             st.markdown("---")
#             st.subheader("Personalized Diet Advice")
#             st.info(advice)




# main_app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import json
import google.generativeai as genai
from huggingface_hub import InferenceClient
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# --- 1. CONFIGURATION & SETUP ---
# It's highly recommended to use relative paths for better portability.
# If your 'data' folder and model file are next to this script, use paths like these:
# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = "data/train"
# NUTRITION_CSV_PATH = 'data/Indian_Food_Nutrition_Processed.csv'

# Using the absolute paths from your setup. Ensure these are correct.
MODEL_PATH = "indian_food_classifier_final.pth"
LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# Image transformation for the food classification model
classification_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. MODEL & DATA LOADING (Cached for performance) ---

@st.cache_resource
def load_food_classifier():
    """Loads the local ResNet18 model by downloading it from Hugging Face Hub."""
    st.info("Downloading food classification model from the Hub...")
    
    # --- THIS IS THE KEY CHANGE ---
    # Define your Hugging Face model repository ID
    # Replace 'YOUR_HF_USERNAME' with your actual Hugging Face username
    HF_REPO_ID = "muku2001/nutri-advisor-classifier"
    MODEL_FILENAME = "indian_food_classifier_final.pth"
    
    # Download the model file from the Hub. It gets cached automatically.
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    except Exception as e:
        st.error(f"Error downloading model from Hugging Face Hub: {e}")
        st.error(f"Please ensure your repository '{HF_REPO_ID}' is public and the filename '{MODEL_FILENAME}' is correct.")
        st.stop()
    # --- END OF CHANGE ---

    # The rest of the function remains the same, but it now loads from the downloaded path
    labels = sorted([d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))])
    
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    
    # Load the state dict from the path provided by the download function
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    st.success("Food classification model loaded successfully.")
    return model, labels

@st.cache_resource
def load_ner_pipeline():
    """Loads the Biomedical NER pipeline from Hugging Face."""
    st.info("Loading Clinical NER model...")
    ner_pipe = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    st.success("Clinical NER model loaded.")
    return ner_pipe

@st.cache_data
def load_and_prepare_nutrition_data():
    """Loads nutrition data and prepares it for the vector database."""
    if not os.path.exists(NUTRITION_CSV_PATH):
        st.warning(f"Nutrition data not found at {NUTRITION_CSV_PATH}. Nutrition lookup will be unavailable.")
        return None
    try:
        df = pd.read_csv(NUTRITION_CSV_PATH)
        df['Dish Name'] = df['Dish Name'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading nutrition CSV: {e}")
        return None

@st.cache_resource
def setup_vector_db(_nutrition_df):
    """Initializes ChromaDB and ingests nutrition data if the collection is empty."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="nutrition_data")
    
    if collection.count() == 0 and _nutrition_df is not None:
        st.info("Setting up nutrition vector database...")
        documents = _nutrition_df['Dish Name'].tolist()
        metadata = _nutrition_df.to_dict('records')
        ids = [f"id_{i}" for i in range(len(documents))]
        collection.add(documents=documents, metadatas=metadata, ids=ids)
        st.success("Nutrition database setup complete.")
        
    return collection

# --- 3. CORE AI PIPELINE FUNCTIONS ---

def classify_food(image: Image.Image, model, labels):
    """Uses the local model to predict the food name from an image."""
    image_tensor = classification_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        prediction_idx = output.argmax(1).item()
        return labels[prediction_idx]

def estimate_portion(image: Image.Image, food_name: str, google_api_key: str):
    """
    ⭐ NEW FUNCTION: Uses Gemini to estimate the portion size.
    """
    try:
        genai.configure(api_key=google_api_key)
        # --- THIS IS THE CORRECTED LINE ---
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        Analyze the image containing '{food_name.replace('_', ' ')}'.
        Your task is to estimate the portion size.

        Provide your answer ONLY in a valid JSON format with the following keys:
        - "estimated_weight_grams": An integer representing the estimated weight in grams.
        - "estimated_pieces": An integer for countable items (like samosas, pooris, idlis). If not applicable, return 1.
        - "confidence": A string (e.g., "High", "Medium", "Low") indicating your confidence in the estimate.
        - "reasoning": A brief sentence explaining your estimation basis (e.g., "Based on standard restaurant serving size.").

        Example Response for an image of two samosas:
        {{
            "estimated_weight_grams": 120,
            "estimated_pieces": 2,
            "confidence": "High",
            "reasoning": "The image shows two standard-sized samosas, each typically weighing around 60 grams."
        }}
        """
        
        response = model.generate_content([prompt, image])
        
        # Clean the response to ensure it's valid JSON
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)

    except Exception as e:
        st.error(f"Error during portion estimation with Gemini: {e}")
        return None
def retrieve_nutrition_info(food_label: str, collection):
    """Retrieves nutrition data from the vector database for a given food label."""
    if collection.count() == 0:
        return None, None
    
    # Clean up the label for better matching
    query_text = food_label.replace('food indian_food ', '').replace('_', ' ')
    
    results = collection.query(query_texts=[query_text], n_results=1)
    
    if results and results['metadatas'][0]:
        retrieved_data = results['metadatas'][0][0]
        matched_name = retrieved_data.get('Dish Name', 'N/A')
        return retrieved_data, matched_name
    return None, None

def analyze_ehr_with_llm(ehr_text: str, ner_pipeline, hf_token: str):
    """A 2-step process to extract and structure clinical data from EHR text."""
    if not ehr_text.strip():
        return {"conditions": [], "symptoms": [], "medications": []}, "No EHR text provided."

    # 1. Run local NER model to pre-identify entities
    ner_entities = ner_pipeline(ehr_text)
    if not ner_entities:
        ner_results_str = "No medical entities identified by the NER model."
    else:
        ner_results_str = "\n".join([f"- {e['word']} (Type: {e['entity_group']})" for e in ner_entities])
    
    # 2. Use LLM (Llama 3) to structure the data, guided by NER results
    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
    prompt = f"""
    Based on the patient record below, and guided by the pre-identified entities, structure the information into a JSON object.
    The JSON must contain three keys: "conditions", "symptoms", and "medications".
    Return ONLY the raw JSON object and nothing else.

    PRE-IDENTIFIED ENTITIES:
    {ner_results_str}

    PATIENT RECORD:
    "{ehr_text}"

    JSON Output:
    """
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=200, temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()
        structured_data = json.loads(raw_text)
        return structured_data, ner_results_str
    except Exception as e:
        st.warning(f"Failed to structure clinical data with LLM. Error: {e}")
        return {"conditions": [], "symptoms": [], "medications": []}, ner_results_str

def generate_dietary_advice(clinical_data, food_name, nutrition_info, portion_info, hf_token: str):
    """Generates the final personalized dietary advice using an LLM."""
    
    # Format all the collected data into a comprehensive prompt
    conditions = ", ".join(clinical_data.get('conditions', [])) or "none"
    symptoms = ", ".join(clinical_data.get('symptoms', [])) or "none"
    portion_desc = f"{portion_info.get('estimated_pieces', 1)} piece(s), approx. {portion_info.get('estimated_weight_grams', 'N/A')}g"
    
    nut_keys = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 'Sodium (mg)']
    nut_desc = ', '.join(f"{k}: {nutrition_info.get(k, 'N/A')}" for k in nut_keys)

    prompt = f"""
    You are an expert dietitian providing personalized advice for a patient in India.

    PATIENT DATA:
    - Known Conditions: {conditions}
    - Reported Symptoms: {symptoms}

    FOOD DATA:
    - Food Item: {food_name}
    - Estimated Portion: {portion_desc}
    - Nutritional Info (per 100g): {nut_desc}

    INSTRUCTIONS:
    Provide concise, practical, and personalized dietary advice in about 4-5 sentences.
    1. Assess if this food, in this portion size, is suitable given the patient's conditions.
    2. Highlight specific nutritional risks (e.g., high sodium for hypertension).
    3. Suggest a clear action (e.g., "Enjoy this portion," "Consider limiting to half," "Best to avoid").
    4. If applicable, recommend one healthier alternative.
    """
    
    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a professional dietitian."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating advice: {e}"

# --- 4. STREAMLIT USER INTERFACE ---

st.set_page_config(layout="wide")
st.title("🥗 Nutri-Advisor: AI Diet Advice with Portion Estimation")
st.markdown("Upload a food image, provide patient context, and get AI-powered dietary recommendations.")

# --- Initial model loading ---
food_model, food_labels = load_food_classifier()
ner_model = load_ner_pipeline()
nutrition_df = load_and_prepare_nutrition_data()
nutrition_db = setup_vector_db(nutrition_df)
# ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Upload Food Image")
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Food Image", use_column_width=True)

with col2:
    st.subheader("Step 2: Provide Context")
    ehr_summary = st.text_area(
        "Patient's Clinical Summary (optional)",
        "Patient has a history of hypertension and type 2 diabetes. Complains of occasional bloating.",
        height=150
    )
    st.subheader("Step 3: API Keys")
    google_api_key = st.text_input("Enter your Google AI API Key", type="password", help="Required for portion estimation.")
    hf_token = st.text_input("Enter your Hugging Face API Token", type="password", help="Required for data structuring and advice generation.")

st.markdown("---")

if st.button("🚀 Analyze and Get Advice", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a food image.")
    elif not google_api_key or not hf_token:
        st.warning("Please enter both Google AI and Hugging Face API keys.")
    else:
        # --- EXECUTE THE FULL AI PIPELINE ---
        
        # 1. Classify Food
        with st.spinner("Analyzing food type..."):
            food_label = classify_food(image, food_model, food_labels)
            food_name = food_label.replace('food indian_food ', '').replace('_', ' ').title()
        st.success(f"**Identified Food:** {food_name}")

        # 2. Estimate Portion
        with st.spinner("Estimating portion size with Gemini Vision..."):
            portion_data = estimate_portion(image, food_name, google_api_key)
        
        if portion_data:
            st.success(f"**Estimated Portion:** Approx. **{portion_data['estimated_weight_grams']}g** ({portion_data['estimated_pieces']} piece(s))")
            with st.expander("View Portion Estimation Details"):
                st.json(portion_data)

        # 3. Retrieve Nutrition
        with st.spinner("Retrieving nutritional information..."):
            nutrition_data, matched_name = retrieve_nutrition_info(food_label, nutrition_db)
        
        if nutrition_data:
            st.success(f"**Nutrition Info Found For:** {matched_name.title()}")
        else:
            st.warning("Could not find specific nutrition data for this food.")

        # 4. Analyze EHR
        with st.spinner("Analyzing clinical summary..."):
            clinical_data, ner_log = analyze_ehr_with_llm(ehr_summary, ner_model, hf_token)

        # 5. Generate Final Advice
        if portion_data and nutrition_data:
            with st.spinner("Generating personalized dietary advice..."):
                final_advice = generate_dietary_advice(clinical_data, matched_name, nutrition_data, portion_data, hf_token)
            
            st.markdown("---")
            st.subheader("💡 Personalized Dietary Advice")
            st.info(final_advice)

            with st.expander("Show Detailed Analysis"):
                st.subheader("Extracted Clinical Information")
                st.json(clinical_data)
                st.subheader("Nutrition Facts (per 100g)")
                st.dataframe(pd.Series(nutrition_data, name="Value"), use_container_width=True)
        else:
            st.error("Could not generate advice because portion estimation or nutrition retrieval failed.")









#  """Loads the local ResNet18 model and infers labels from the folder structure."""
#     st.info("Loading local food classification model...")
#     if not os.path.exists(LABELS_DIR):
#         st.error(f"Labels directory not found at {LABELS_DIR}. Please check the path.")
#         st.stop()
    
#     # This logic correctly reads your sub-folders as class labels
#     labels = sorted([d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))])
    
#     if not labels:
#         st.error(f"No food sub-folders found in {LABELS_DIR}. Cannot determine class labels.")
#         st.stop()

#     model = models.resnet18(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is in the correct location.")
#         st.stop()
        
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     st.success("Food classification model loaded.")
#     return model, labels