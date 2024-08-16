# autogpt/materials_project_api.py

import requests
import spacy

nlp = spacy.load("en_core_web_sm")

def is_material_science_query(query: str) -> bool:
    material_keywords = ["material", "element", "compound", "alloy", "crystal", "structure", 
                         "properties", "band gap", "density", "elasticity", "conductivity", 
                         "melting point", "boiling point", "atomic", "molecular"]
    doc = nlp(query.lower())
    return any(token.text in material_keywords for token in doc)

def fetch_material_data(material_id, query_type, api_key):
    base_url = "https://materialsproject.org/rest/v2/materials"
    endpoints = {
        "band_gap": f"{base_url}/{material_id}/vasp",
        "density": f"{base_url}/{material_id}/density",
        "elasticity": f"{base_url}/{material_id}/elasticity",
    }
    url = endpoints.get(query_type)
    if not url:
        return None
    headers = {"X-API-KEY": api_key}
    response = requests.get(url, headers=headers)
    return response.json()

def process_material_query(query: str, api_key: str) -> str:
    if not is_material_science_query(query):
        return "This query doesn't appear to be related to material science."
    
    material_id = query.split()[-1]  # Simplified extraction, improve as needed
    if "band gap" in query.lower():
        query_type = "band_gap"
    elif "density" in query.lower():
        query_type = "density"
    elif "elasticity" in query.lower():
        query_type = "elasticity"
    else:
        return "Unable to determine the type of material property requested."

    data = fetch_material_data(material_id, query_type, api_key)
    if data and "response" in data:
        if query_type == "band_gap":
            band_gap = data["response"]["property"].get("band_gap", "N/A")
            return f"The band gap of the material {material_id} is {band_gap} eV."
        elif query_type == "density":
            density = data["response"]["property"].get("density", "N/A")
            return f"The density of the material {material_id} is {density} g/cm³."
        elif query_type == "elasticity":
            elasticity = data["response"]["property"].get("elasticity", "N/A")
            return f"The elasticity of the material {material_id} is {elasticity} GPa."
    else:
        return f"Unable to fetch data for material {material_id}."