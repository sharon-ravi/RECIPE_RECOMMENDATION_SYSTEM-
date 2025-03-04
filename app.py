from pymongo import MongoClient
import spacy
import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import re

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .recipe-card {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        background-color: #ffffff;
        margin-bottom: 10px;
    }
    .recipe-header {
        font-weight: bold;
        font-size: 18px;
        color: #333;
    }
    .subheader {
        font-size: 16px;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Connect to MongoDB and load data
client = MongoClient("mongodb://localhost:27017/")
db = client["s1"]  # Replace with your database name
recipes_collection = db["recipe"]  # Replace with your collection name

# Load recipes data into a DataFrame
recipes_data = pd.DataFrame(list(recipes_collection.find({}, {"_id": 0, "recipe_name": 1, "recipe_type": 1, "ingredients": 1, "instructions": 1})))

# Check if data is available for training
if len(recipes_data) < 10:
    st.warning("Not enough data to train the model. Please add more recipes to the database.")
else:
    # Prepare data for training
    recipes_data['text'] = recipes_data['recipe_name'] + " " + recipes_data['ingredients'].apply(lambda x: " ".join(x))
    X = recipes_data['text']
    y = recipes_data['recipe_type']
    
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #  Train a Naive Bayes model using TF-IDF features
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model trained with accuracy: {accuracy:.2f}")

nlp = None
EDAMAM_APP_ID = '3b101cee'  # Replace with your API ID
EDAMAM_APP_KEY = '55939d19017e348c2a2be920b0a526fc'  # Replace with your API Key

# Function to classify recipe type using the trained ML model
def classify_recipe_type(user_input):
    predicted_type = model.predict([user_input])[0]
    return predicted_type

# Step 7: Function to detect negation and excluded ingredients
def extract_negations(user_query):
    pattern = r"\b(?:without|don['’]t|do\s+not|not|never|exclude|no|without)\s+([\w\s,]+)"
    negations = re.findall(pattern, user_query)
    excluded_ingredients = []
    for negation in negations:
        excluded_ingredients.extend([ingredient.strip() for ingredient in negation.split(',')])
    return list(set(excluded_ingredients))

# Step 8: Function to process user query and fetch related recipes with chunk extraction
def process_user_query(user_query):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_md")
    doc = nlp(user_query)

    # Extract entities, POS tags, and chunks
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    entity_texts = [ent[0] for ent in entities]
    combined_tags = list(set(keywords + entity_texts))
    chunks = [chunk.text for chunk in doc.noun_chunks]

    # Extract excluded ingredients based on negations
    excluded_ingredients = extract_negations(user_query)
    main_recipe = next((keyword for keyword in keywords if keyword.lower() not in excluded_ingredients), None)
    if "thirst" in user_query.lower() or "killing" in user_query.lower():
        if "chicken" in user_query.lower():
            main_recipe = "chicken"  
    query_filter = {
        "$or": [
            {"recipe_name": {"$regex": "|".join(combined_tags), "$options": "i"}},
            {"ingredients": {"$regex": "|".join(combined_tags), "$options": "i"}},
            {"tags": {"$regex": "|".join(combined_tags), "$options": "i"}}
        ]
    }
    if excluded_ingredients:
        query_filter["ingredients"] = {"$not": {"$regex": "|".join(excluded_ingredients), "$options": "i"}}

    if main_recipe:
        query_filter["recipe_name"] = {"$regex": main_recipe, "$options": "i"}

    matching_recipes = list(recipes_collection.find(query_filter))

    if not matching_recipes:
        api_recipes = fetch_recipes_from_api(combined_tags, excluded_ingredients)
        return {"recipes": api_recipes, "entities": entities, "pos_tags": pos_tags, "chunks": chunks}

    return {"recipes": matching_recipes, "entities": entities, "pos_tags": pos_tags, "chunks": chunks}

# Step 9: Function to fetch recipes from Edamam API with negation handling
def fetch_recipes_from_api(tags, excluded_ingredients):
    search_query = "+".join(tags)
    url = f"https://api.edamam.com/search?q={search_query}&app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        recipes = []
        for hit in data.get("hits", []):
            recipe = hit["recipe"]
            if any(excluded_ingredient.lower() in recipe["ingredientLines"] for excluded_ingredient in excluded_ingredients):
                continue
            recipes.append({
                "recipe_name": recipe["label"],
                "ingredients": recipe["ingredientLines"],
                "tags": tags,
                "instructions": "Instructions not available from API"
            })
        return recipes
    else:
        return []

st.title("Recipe Recommendation System ")

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "selected_recipe_name" not in st.session_state:
    st.session_state.selected_recipe_name = None

user_query = st.text_input("Enter your query:")

if st.button("Search Recipe") and user_query:
    st.session_state.query_result = process_user_query(user_query)
    st.session_state.selected_recipe_name = None

if st.session_state.query_result:
    query_result = st.session_state.query_result
    recipes = query_result["recipes"]
    entities = query_result["entities"]
    pos_tags = query_result["pos_tags"]
    chunks = query_result["chunks"]

    predicted_type = classify_recipe_type(user_query)

    if recipes:
        st.subheader("Matched Recipes")
        recipe_names = [recipe["recipe_name"] for recipe in recipes]

        st.session_state.selected_recipe_name = st.selectbox(
            "Select a recipe:",
            recipe_names,
            index=recipe_names.index(st.session_state.selected_recipe_name) if st.session_state.selected_recipe_name else 0
        )
        
        selected_recipe = next((recipe for recipe in recipes if recipe["recipe_name"] == st.session_state.selected_recipe_name), None)
        
        if selected_recipe:
            st.markdown(f"<div class='recipe-card'><div class='recipe-header'>{selected_recipe['recipe_name']}</div>", unsafe_allow_html=True)
            st.markdown(f"*Ingredients:* {', '.join(selected_recipe['ingredients'])}")
            st.markdown(f"*Tags:* {', '.join(selected_recipe.get('tags', []))}")

            if "instructions" in selected_recipe and selected_recipe["instructions"] != "Instructions not available from API":
                st.markdown(f"*Instructions:* {selected_recipe['instructions']}")
            else:
                st.markdown("Instructions not available for this recipe.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("### Extracted Entities:")
        formatted_entities = ", ".join([f"{text}: {label}" for text, label in entities])
        st.write(formatted_entities if formatted_entities else "No entities found.")

        st.write("### Parts of Speech Tags:")
        formatted_pos_tags = ", ".join([f"{text}: {pos}" for text, pos in pos_tags])
        st.write(formatted_pos_tags if formatted_pos_tags else "No POS tags found.")

   
        st.write("### Noun Chunks:")
        formatted_chunks = ", ".join(chunks)
        st.write(formatted_chunks if formatted_chunks else "No noun chunks found.")
        st.write(f"Predicted Recipe Type: {predicted_type}")


    else:
        st.write("No matching recipes found. Try a different query.")
