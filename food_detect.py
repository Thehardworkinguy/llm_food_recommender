# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pandas as pd
# import difflib


# # Configurations

# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
# NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Load model & labels

# @st.cache_resource
# def load_model_and_labels():
#     labels = sorted(os.listdir(LABELS_DIR))
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, labels

# model, labels = load_model_and_labels()


# # Load nutrition data

# @st.cache_data
# def load_nutrition_data():
#     df = pd.read_csv(NUTRITION_CSV_PATH)
#     return df

# nutrition_df = load_nutrition_data()


# # Image preprocessing

# IMG_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # Clean and robust nutrition lookup with debug info

# def clean_food_name(label):
#     label = label.replace('food', '').replace('indian_food', '').replace('_', ' ')
#     words = label.lower().split()
#     food_name = words[-1] 
#     compound_name = ' '.join(words[-2:]) if len(words) >= 2 else food_name
#     return food_name, compound_name

# def lookup_nutrition(predicted_label):
#     food_name, compound_name = clean_food_name(predicted_label)
#     dish_names = nutrition_df['Dish Name'].astype(str).str.lower().tolist()
#     # Print search details for debugging
#     print(f"[DEBUG] Looking for: '{food_name}' or '{compound_name}' in dish names ... ")

#     # 1. Substring Match (both single and compound)
#     for name in dish_names:
#         if food_name in name or compound_name in name:
#             found_idx = dish_names.index(name)
#             print(f"[DEBUG] Substring match found: {name}")
#             return nutrition_df.iloc[found_idx].to_dict(), name

#     # 2. Fuzzy Match
#     matches = difflib.get_close_matches(food_name, dish_names, n=1, cutoff=0.6)
#     if not matches and compound_name != food_name:
#         matches = difflib.get_close_matches(compound_name, dish_names, n=1, cutoff=0.6)
#     if matches:
#         found_idx = dish_names.index(matches[0])
#         print(f"[DEBUG] Fuzzy match found: {matches}")
#         return nutrition_df.iloc[found_idx].to_dict(), matches

#     print("[DEBUG] No match found")
#     return None, None


# # Prediction function

# def predict(image: Image.Image):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         pred_idx = outputs.argmax(dim=1).item()
#     return labels[pred_idx]


# # STREAMLIT UI

# st.title("Indian Food Image Classifier with Nutrition Lookup")

# uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Predict Food Item"):
#         prediction = predict(image)
#         st.success(f"Predicted food item: **{prediction}**")
        
#         nutrition, matched_name = lookup_nutrition(prediction)
#         if nutrition:
#             st.info(f"Matched nutrition entry: **{matched_name.title()}**")
#             st.subheader("Nutrition Facts (per 100g):")
#             st.write(f"Calories: {nutrition.get('Calories (kcal)', 'N/A')} kcal")
#             st.write(f"Carbohydrate: {nutrition.get('Carbohydrate (g)', 'N/A')} g")
#             st.write(f"Protein: {nutrition.get('Protein (g)', 'N/A')} g")
#             st.write(f"Fat: {nutrition.get('Fats (g)', 'N/A')} g")
#             st.write(f"Free Sugar: {nutrition.get('Free Sugar (g)', 'N/A')} g")
#             st.write(f"Fibre: {nutrition.get('Fibre (g)', 'N/A')} g")
#             st.write(f"Sodium: {nutrition.get('Sodium (mg)', 'N/A')} mg")
#             st.write(f"Calcium: {nutrition.get('Calcium (mg)', 'N/A')} mg")
#             st.write(f"Iron: {nutrition.get('Iron (mg)', 'N/A')} mg")
#             st.write(f"Vitamin C: {nutrition.get('Vitamin C (mg)', 'N/A')} mg")
#             st.write(f"Folate: {nutrition.get('Folate (µg)', 'N/A')} µg")
#         else:
#             st.warning("Nutrition information not found for this food item.")









# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pandas as pd
# import difflib
# import openai

# # Configurations
# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
# NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @st.cache_resource
# def load_model_and_labels():
#     labels = sorted(os.listdir(LABELS_DIR))
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, labels

# model, labels = load_model_and_labels()

# @st.cache_data
# def load_nutrition_data():
#     df = pd.read_csv(NUTRITION_CSV_PATH)
#     df['Dish Name Lower'] = df['Dish Name'].astype(str).str.lower()
#     return df

# nutrition_df = load_nutrition_data()

# IMG_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# def clean_food_name(label):
#     label = label.replace('food', '').replace('indian_food', '').replace('_', ' ')
#     words = label.lower().split()
#     food_name = words[-1]
#     compound_name = ' '.join(words[-2:]) if len(words) >= 2 else food_name
#     return food_name, compound_name

# def lookup_nutrition(predicted_label):
#     food_name, compound_name = clean_food_name(predicted_label)
#     dish_names = nutrition_df['Dish Name Lower'].tolist()
#     st.experimental_write(f"**[DEBUG]** Searching nutrition entries for: '{food_name}' or '{compound_name}'")

#     for idx, name in enumerate(dish_names):
#         if food_name in name or compound_name in name:
#             st.experimental_write(f"**[DEBUG]** Substring match found: {name.title()}")
#             return nutrition_df.iloc[idx].to_dict(), name

#     matches = difflib.get_close_matches(food_name, dish_names, n=1, cutoff=0.6)
#     if not matches and compound_name != food_name:
#         matches = difflib.get_close_matches(compound_name, dish_names, n=1, cutoff=0.6)
#     if matches:
#         idx = dish_names.index(matches[0])
#         st.experimental_write(f"**[DEBUG]** Fuzzy match found: {matches[0].title()}")
#         return nutrition_df.iloc[idx].to_dict(), matches[0]

#     st.experimental_write("**[DEBUG]** No nutrition information found for this item.")
#     return None, None

# def predict(image: Image.Image):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         pred_idx = outputs.argmax(dim=1).item()
#     return labels[pred_idx]

# # Extract medical conditions from EHR text (keyword-based)
# def extract_conditions(ehr_text):
#     condition_keywords = {
#         'diabetes': ['diabetes', 'hyperglycemia', 'high blood sugar'],
#         'hypertension': ['hypertension', 'high blood pressure', 'bp'],
#         'chronic kidney disease': ['chronic kidney disease', 'ckd', 'renal failure'],
#         'anemia': ['anemia', 'low hemoglobin'],
#         'coronary artery disease': ['coronary artery disease', 'cad', 'ischemia']
#     }
#     found_conditions = []
#     ehr_text_lower = ehr_text.lower()
#     for condition, keywords in condition_keywords.items():
#         if any(kw in ehr_text_lower for kw in keywords):
#             found_conditions.append(condition)
#     return found_conditions

# def construct_prompt(ehr_text, conditions, food_name, nutrition_info):
#     nutritional_details = ', '.join(f"{k}: {v}" for k, v in nutrition_info.items() if k not in ['Dish Name', 'Dish Name Lower'])
#     conditions_desc = ', '.join(conditions) if conditions else "no known chronic conditions"
#     prompt = (
#         f"Patient Electronic Health Record Summary:\n{ehr_text}\n\n"
#         f"Detected or reported conditions: {conditions_desc}.\n\n"
#         f"The patient plans to consume the food item: {food_name}.\n"
#         f"Nutrient breakdown for one serving is: {nutritional_details}.\n\n"
#         "As a clinical dietitian, provide personalized dietary advice "
#         "for this patient considering their health status. Mention any potential risks, "
#         "contraindications, or recommendations specific to the patient's condition. "
#         "Suggest healthier alternatives if appropriate. Keep the advice clear and actionable."
#     )
#     return prompt

# def query_openai_chat(prompt, api_key):
#     openai.api_key = api_key
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful and knowledgeable clinical dietitian assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.6,
#             max_tokens=400,
#             n=1,
#             stop=None,
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         return f"OpenAI API Error: {str(e)}"

# # Streamlit UI
# st.title("Indian Food Classifier + Personalized Dietary Advice")

# uploaded_file = st.file_uploader("Upload Indian food image", type=["jpg", "jpeg", "png"])

# ehr_text = st.text_area("Paste patient's health record or EHR summary here")

# api_key = st.text_input("Enter OpenAI API Key", type="password")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Food Image", use_column_width=True)
    
#     if st.button("Predict Food and Get Dietary Advice"):
#         with st.spinner("Running image classification and generating dietary advice..."):
#             predicted_label = predict(image)
#             st.success(f"Predicted Food Item: **{predicted_label}**")
            
#             nutrition_info, matched_name = lookup_nutrition(predicted_label)
            
#             if nutrition_info is None:
#                 st.warning("Nutrition information not found for the predicted food.")
#             else:
#                 st.info(f"Matched Nutrition Entry: **{matched_name.title()}**")
#                 st.write("### Nutrition Facts (per 100g)")
#                 keys_to_show = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 
#                                 'Free Sugar (g)', 'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 
#                                 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
#                 for key in keys_to_show:
#                     val = nutrition_info.get(key, 'N/A')
#                     st.write(f"- **{key}**: {val}")
                
#                 if not ehr_text.strip():
#                     st.warning("Please provide patient's health record text for personalized advice.")
#                 elif not api_key:
#                     st.warning("Please enter your OpenAI API key to get personalized advice.")
#                 else:
#                     conditions = extract_conditions(ehr_text)
#                     prompt = construct_prompt(ehr_text, conditions, matched_name.title(), nutrition_info)
#                     st.subheader("LLM Prompt Preview:")
#                     st.code(prompt, language='text')
                    
#                     advice = query_openai_chat(prompt, api_key)
#                     st.subheader("Personalized Dietary Advice from LLM:")
#                     st.write(advice)


# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pandas as pd
# import difflib
# import openai

# # Configurations
# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
# NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @st.cache_resource
# def load_model_and_labels():
#     if not os.path.exists(LABELS_DIR):
#         raise FileNotFoundError(f"Labels directory not found: {LABELS_DIR}")
#     labels = [d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))]
#     labels = sorted(labels)
#     if not labels:
#         raise RuntimeError(f"No subdirectories found in labels directory: {LABELS_DIR}")
#     st.write(f"Found labels: {labels}")  # Debug info
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, labels

# model, labels = load_model_and_labels()

# @st.cache_data
# def load_nutrition_data():
#     df = pd.read_csv(NUTRITION_CSV_PATH)
#     df['Dish Name Lower'] = df['Dish Name'].astype(str).str.lower()
#     return df

# nutrition_df = load_nutrition_data()

# IMG_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# def clean_food_name(label):
#     label = label.replace('food', '').replace('indian_food', '').replace('_', ' ')
#     words = label.lower().split()
#     food_name = words[-1]
#     compound_name = ' '.join(words[-2:]) if len(words) >= 2 else food_name
#     return food_name, compound_name

# def lookup_nutrition(predicted_label):
#     food_name, compound_name = clean_food_name(predicted_label)
#     dish_names = nutrition_df['Dish Name'].astype(str).str.lower().tolist()
#     st.write(f"**[DEBUG]** Looking for: '{food_name}' or '{compound_name}' in dish names ...")

#     for name in dish_names:
#         if food_name in name or compound_name in name:
#             st.write(f"**[DEBUG]** Substring match found: {name.title()}")
#             return nutrition_df[nutrition_df['Dish Name'].str.lower() == name].iloc[0].to_dict(), name

#     matches = difflib.get_close_matches(food_name, dish_names, n=1, cutoff=0.6)
#     if not matches and compound_name != food_name:
#         matches = difflib.get_close_matches(compound_name, dish_names, n=1, cutoff=0.6)
#     if matches:
#         st.write(f"**[DEBUG]** Fuzzy match found: {matches}")
#         matched_name = matches[0]
#         return nutrition_df[nutrition_df['Dish Name'].str.lower() == matched_name].iloc[0].to_dict(), matched_name

#     st.write("**[DEBUG]** No match found")
#     return None, None

# def predict(image: Image.Image):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         pred_idx = outputs.argmax(dim=1).item()
#     return labels[pred_idx]

# def extract_conditions(ehr_text):
#     condition_keywords = {
#         'diabetes': ['diabetes', 'hyperglycemia', 'high blood sugar'],
#         'hypertension': ['hypertension', 'high blood pressure', 'bp'],
#         'chronic kidney disease': ['chronic kidney disease', 'ckd', 'renal failure'],
#         'anemia': ['anemia', 'low hemoglobin'],
#         'coronary artery disease': ['coronary artery disease', 'cad', 'ischemia']
#     }
#     found_conditions = []
#     ehr_text_lower = ehr_text.lower()
#     for condition, keywords in condition_keywords.items():
#         if any(kw in ehr_text_lower for kw in keywords):
#             found_conditions.append(condition)
#     return found_conditions

# def construct_prompt(ehr_text, conditions, food_name, nutrition_info):
#     nutritional_details = ', '.join(f"{k}: {v}" for k, v in nutrition_info.items() if k not in ['Dish Name', 'Dish Name Lower'])
#     conditions_desc = ', '.join(conditions) if conditions else "no known chronic conditions"
#     prompt = (
#         f"Patient Electronic Health Record Summary:\n{ehr_text}\n\n"
#         f"Detected or reported conditions: {conditions_desc}.\n\n"
#         f"The patient plans to consume the food item: {food_name}.\n"
#         f"Nutrient breakdown for one serving is: {nutritional_details}.\n\n"
#         "As a clinical dietitian, provide personalized dietary advice "
#         "for this patient considering their health status. Mention any potential risks, "
#         "contraindications, or recommendations specific to the patient's condition. "
#         "Suggest healthier alternatives if appropriate. Keep the advice clear and actionable."
#     )
#     return prompt

# # --------- OpenAI v1.x Chat API ---------
# def query_openai_chat(prompt, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful and knowledgeable clinical dietitian assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.6,
#             max_tokens=400,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"OpenAI API Error: {str(e)}"
# # ----------------------------------------

# st.title("Indian Food Classifier + Personalized Dietary Advice")

# uploaded_file = st.file_uploader("Upload Indian food image", type=["jpg", "jpeg", "png"])
# ehr_text = st.text_area("Paste patient's health record or EHR summary here")
# api_key = st.text_input("Enter OpenAI API Key", type="password")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Food Image", use_column_width=True)
    
#     if st.button("Predict Food and Get Dietary Advice"):
#         with st.spinner("Running image classification and generating dietary advice..."):
#             predicted_label = predict(image)
#             st.success(f"Predicted Food Item: **{predicted_label}**")
            
#             nutrition_info, matched_name = lookup_nutrition(predicted_label)
            
#             if nutrition_info is None:
#                 st.warning("Nutrition information not found for the predicted food.")
#             else:
#                 st.info(f"Matched Nutrition Entry: **{matched_name.title()}**")
#                 st.write("### Nutrition Facts (per 100g)")
#                 keys_to_show = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 
#                                 'Free Sugar (g)', 'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 
#                                 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
#                 for key in keys_to_show:
#                     val = nutrition_info.get(key, 'N/A')
#                     st.write(f"- **{key}**: {val}")
                
#                 if not ehr_text.strip():
#                     st.warning("Please provide patient's health record text for personalized advice.")
#                 elif not api_key:
#                     st.warning("Please enter your OpenAI API key to get personalized advice.")
#                 else:
#                     conditions = extract_conditions(ehr_text)
#                     prompt = construct_prompt(ehr_text, conditions, matched_name.title(), nutrition_info)
#                     st.subheader("LLM Prompt Preview:")
#                     st.code(prompt, language='text')
                    
#                     advice = query_openai_chat(prompt, api_key)
#                     st.subheader("Personalized Dietary Advice from LLM:")
#                     st.write(advice)



# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pandas as pd
# import difflib
# import requests

# # Paths (CHANGE these to your actual file locations)
# MODEL_PATH = "indian_food_classifier.pth"
# LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
# NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @st.cache_resource
# def load_model_and_labels():
#     if not os.path.exists(LABELS_DIR):
#         raise FileNotFoundError(f"Labels folder not found at {LABELS_DIR}")
#     labels = [d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))]
#     labels = sorted(labels)
#     if len(labels) == 0:
#         raise RuntimeError("No label folders found in train directory")
#     st.write(f"Detected Labels: {labels}")
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, labels

# model, labels = load_model_and_labels()

# @st.cache_data
# def load_nutrition_data():
#     df = pd.read_csv(NUTRITION_CSV_PATH)
#     df['Dish Name Lower'] = df['Dish Name'].astype(str).str.lower()
#     return df

# nutrition_df = load_nutrition_data()

# IMG_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# def clean_food_name(label):
#     label = label.replace('food', '').replace('indian_food', '').replace('_', ' ')
#     words = label.lower().split()
#     food_name = words[-1]
#     compound_name = ' '.join(words[-2:]) if len(words) >= 2 else food_name
#     return food_name, compound_name

# def lookup_nutrition(predicted_label):
#     food_name, compound_name = clean_food_name(predicted_label)
#     dish_names = nutrition_df['Dish Name Lower'].tolist()
#     st.write(f"Searching nutrition for '{food_name}' or '{compound_name}'...")

#     # Direct substring matches
#     for idx, name in enumerate(dish_names):
#         if food_name in name or compound_name in name:
#             st.write(f"Nutrition matched via substring: {name.title()}")
#             return nutrition_df.iloc[idx].to_dict(), name
    
#     # Fuzzy matching
#     matches = difflib.get_close_matches(food_name, dish_names, n=1, cutoff=0.6)
#     if not matches and compound_name != food_name:
#         matches = difflib.get_close_matches(compound_name, dish_names, n=1, cutoff=0.6)
#     if matches:
#         st.write(f"Nutrition matched via fuzzy: {matches[0].title()}")
#         idx = dish_names.index(matches[0])
#         return nutrition_df.iloc[idx].to_dict(), matches[0]

#     st.write("No matching nutrition found")
#     return None, None

# def predict(image: Image.Image):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image)
#         return labels[output.argmax(1).item()]

# def extract_conditions(ehr_text):
#     keywords_map = {
#         'diabetes': ['diabetes', 'high blood sugar', 'hyperglycemia'],
#         'hypertension': ['hypertension', 'high blood pressure', 'bp'],
#         'chronic kidney disease': ['chronic kidney disease', 'ckd', 'renal failure'],
#         'anemia': ['anemia', 'low hemoglobin'],
#         'coronary artery disease': ['coronary artery disease', 'cad', 'ischemia'],
#     }
#     lower_text = ehr_text.lower()
#     found_conditions = []
#     for cond, keys in keywords_map.items():
#         if any(k in lower_text for k in keys):
#             found_conditions.append(cond)
#     return found_conditions

# def construct_prompt(ehr_text, conditions, food_name, nutrition_info):
#     nut_desc = ', '.join(f"{k}: {v}" for k, v in nutrition_info.items() if k not in ['Dish Name', 'Dish Name Lower'])
#     conditions_desc = ', '.join(conditions) if conditions else "no known conditions"
#     # Zephyr and similar models perform best if you add instruction at start
#     return f"""<s> [INST] You are a professional dietitian helping a patient with the following clinical data:

# Patient summary:
# {ehr_text}

# Known conditions:
# {conditions_desc}

# The patient plans to consume:
# {food_name}

# Nutritional info (per serving):
# {nut_desc}

# Give personalized dietary advice considering the patient's conditions. Highlight risks, benefits, and alternatives clearly. [/INST]
# """

# def query_hf_chat(prompt, hf_token):
#     API_URL = "https://api-inference.huggingface.co/models/gpt2"
#     headers = {"Authorization": f"Bearer {hf_token}"}
#     payload = {
#         "inputs": prompt,
#         "parameters": {"max_new_tokens": 256, "temperature": 0.7},
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
#     if response.status_code == 200:
#         resp_json = response.json()
#         if isinstance(resp_json, dict) and 'error' in resp_json:
#             return f"API Error: {resp_json['error']}"
#         return resp_json[0]['generated_text']
#     elif response.status_code == 503:
#         return "Model is loading (cold start) or busy. Please try again in a minute."
#     elif response.status_code == 401:
#         return "Invalid Hugging Face token. Please check your token."
#     elif response.status_code == 403:
#         return "You don't have access to this model endpoint on Hugging Face."
#     else:
#         return f"HTTP {response.status_code} error: {response.text}"

# st.title("Indian Food Image Classifier + Personalized Diet Advice")

# uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])
# ehr_text = st.text_area("Paste patient's EHR summary here")
# hf_token = st.text_input("Enter your Hugging Face API Token", type="password")

# if uploaded_file:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     if st.button("Classify and Get Dietary Advice"):
#         with st.spinner("Predicting and generating advice..."):
#             pred_label = predict(img)
#             st.success(f"Predicted Food: **{pred_label}**")

#             nutrition, matched_name = lookup_nutrition(pred_label)
#             if not nutrition:
#                 st.warning("No nutrition data found for predicted food.")
#             else:
#                 st.info(f"Nutrition info matched: **{matched_name.title()}**")
#                 st.write("### Nutrition facts (per 100g)")
#                 keys = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 'Free Sugar (g)',
#                         'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
#                 for k in keys:
#                     st.write(f"- {k}: {nutrition.get(k, 'N/A')}")

#                 if not ehr_text.strip():
#                     st.warning("Please enter patient's EHR summary for personalized advice.")
#                 elif not hf_token.strip():
#                     st.warning("Please enter your Hugging Face API token.")
#                 else:
#                     conds = extract_conditions(ehr_text)
#                     prompt = construct_prompt(ehr_text, conds, matched_name.title(), nutrition)
#                     st.subheader("Prompt sent to Hugging Face API:")
#                     st.code(prompt)

#                     advice = query_hf_chat(prompt, hf_token)
#                     st.subheader("Personalized Diet Advice:")
#                     st.write(advice)


import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import difflib
from huggingface_hub import InferenceClient  # ✅ official client

# Paths (CHANGE these to your actual file locations)
MODEL_PATH = "indian_food_classifier.pth"
LABELS_DIR = r"C:\Users\mukun\Downloads\01ykn2lflluk39dsqhi4bub\images.cv_01ykn2lflluk39dsqhi4bub\data\train"
NUTRITION_CSV_PATH = r'C:\Users\mukun\Downloads\Indian_Food_Nutrition_Processed.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(LABELS_DIR):
        raise FileNotFoundError(f"Labels folder not found at {LABELS_DIR}")
    labels = [d for d in os.listdir(LABELS_DIR) if os.path.isdir(os.path.join(LABELS_DIR, d))]
    labels = sorted(labels)
    if len(labels) == 0:
        raise RuntimeError("No label folders found in train directory")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, labels

model, labels = load_model_and_labels()

@st.cache_data
def load_nutrition_data():
    df = pd.read_csv(NUTRITION_CSV_PATH)
    df['Dish Name Lower'] = df['Dish Name'].astype(str).str.lower()
    return df

nutrition_df = load_nutrition_data()

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def clean_food_name(label):
    label = label.replace('food', '').replace('indian_food', '').replace('_', ' ')
    words = label.lower().split()
    food_name = words[-1]
    compound_name = ' '.join(words[-2:]) if len(words) >= 2 else food_name
    return food_name, compound_name

def lookup_nutrition(predicted_label):
    food_name, compound_name = clean_food_name(predicted_label)
    dish_names = nutrition_df['Dish Name Lower'].tolist()

    # Direct substring matches
    for idx, name in enumerate(dish_names):
        if food_name in name or compound_name in name:
            return nutrition_df.iloc[idx].to_dict(), name
    
    # Fuzzy matching
    matches = difflib.get_close_matches(food_name, dish_names, n=1, cutoff=0.6)
    if not matches and compound_name != food_name:
        matches = difflib.get_close_matches(compound_name, dish_names, n=1, cutoff=0.6)
    if matches:
        idx = dish_names.index(matches[0])
        return nutrition_df.iloc[idx].to_dict(), matches[0]

    return None, None

def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        return labels[output.argmax(1).item()]

def extract_conditions(ehr_text):
    keywords_map = {
        'diabetes': ['diabetes', 'high blood sugar', 'hyperglycemia'],
        'hypertension': ['hypertension', 'high blood pressure', 'bp'],
        'chronic kidney disease': ['chronic kidney disease', 'ckd', 'renal failure'],
        'anemia': ['anemia', 'low hemoglobin'],
        'coronary artery disease': ['coronary artery disease', 'cad', 'ischemia'],
    }
    lower_text = ehr_text.lower()
    found_conditions = []
    for cond, keys in keywords_map.items():
        if any(k in lower_text for k in keys):
            found_conditions.append(cond)
    return found_conditions

def construct_prompt(ehr_text, conditions, food_name, nutrition_info):
    nut_desc = ', '.join(f"{k}: {v}" for k, v in nutrition_info.items() if k not in ['Dish Name', 'Dish Name Lower'])
    conditions_desc = ', '.join(conditions) if conditions else "no known conditions"
    return f"""<s>[INST] You are a professional dietitian helping a patient with the following clinical data:

Patient summary:
{ehr_text}

Known conditions:
{conditions_desc}

The patient plans to consume:
{food_name}

Nutritional info (per serving):
{nut_desc}

Give personalized dietary advice considering the patient's conditions. Highlight risks, benefits, and alternatives clearly. [/INST]"""

# ✅ Hugging Face API via InferenceClient
def query_hf_chat(prompt, hf_token):
    try:
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a professional dietitian."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.7
        )

        # Get the actual text reply
        return response.choices[0].message["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# ------------------- Streamlit UI -------------------

st.title("Indian Food Image Classifier + Personalized Diet Advice")

uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])
ehr_text = st.text_area("Paste patient's EHR summary here")
hf_token = st.text_input("Enter your Hugging Face API Token", type="password")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify and Get Dietary Advice"):
        with st.spinner("Predicting and generating advice..."):
            pred_label = predict(img)
            st.success(f"Predicted Food: **{pred_label}**")

            nutrition, matched_name = lookup_nutrition(pred_label)
            if not nutrition:
                st.warning("No nutrition data found for predicted food.")
            else:
                st.info(f"Nutrition info matched: **{matched_name.title()}**")
                st.write("### Nutrition facts (per 100g)")
                keys = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fats (g)', 'Free Sugar (g)',
                        'Fibre (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)']
                for k in keys:
                    st.write(f"- {k}: {nutrition.get(k, 'N/A')}")

                if not ehr_text.strip():
                    st.warning("Please enter patient's EHR summary for personalized advice.")
                elif not hf_token.strip():
                    st.warning("Please enter your Hugging Face API token.")
                else:
                    conds = extract_conditions(ehr_text)
                    prompt = construct_prompt(ehr_text, conds, matched_name.title(), nutrition)
                    st.subheader("Prompt sent to Hugging Face API:")
                    st.code(prompt)

                    advice = query_hf_chat(prompt, hf_token)
                    st.subheader("Personalized Diet Advice:")
                    st.write(advice)
