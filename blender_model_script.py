import chainlit as cl
import openai
import requests
import os
import zipfile
import re
import logging

# Configuration
SKETCHFAB_API_TOKEN = "4d62a3a564334de8a2c97bd20bfd25f3"
CONTENT_DIR = "C:/Users/kalaivani/Desktop/Genai/animation/content/sketchfab_model"
AZURE_ENDPOINT = "https://ailearning8809288724.openai.azure.com/"
AZURE_API_KEY = "api_key"
AZURE_API_VERSION = "2024-02-15-preview"
DEPLOYMENT_NAME = "gpt-4o"

# Initialize Azure OpenAI client
openai.azure_endpoint = AZURE_ENDPOINT
openai.api_key = AZURE_API_KEY
openai.api_version = AZURE_API_VERSION
# Configure logging
logging.basicConfig(level=logging.INFO)

def find_local_model(prompt):
    """Search for a model file in the content directory matching the prompt."""
    extract_dir = os.path.join(CONTENT_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith((".glb", ".gltf", ".fbx")) and prompt.lower() in file.lower():
                logging.info(f"Found local model: {os.path.join(root, file)}")
                return os.path.join(root, file)
    logging.info(f"No local model found for prompt: {prompt}")
    return None

def sanitize_filename(name):
    """Remove invalid characters from a filename."""
    sanitized = re.sub(r'[^\w\-_\. ]', '', name).strip()
    if not sanitized:
        return "default_model"
    return sanitized

def search_and_download_model(prompt):
    """Search and download a model from Sketchfab, naming it based on prompt."""
    os.makedirs(CONTENT_DIR, exist_ok=True)
    extract_dir = os.path.join(CONTENT_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    search_url = "https://api.sketchfab.com/v3/search"
    headers = {"Authorization": f"Token {SKETCHFAB_API_TOKEN}"}
    params = {
        "q": prompt,
        "type": "models",
        "downloadable": True,
        "sort_by": "-relevance",  # Sort by relevance to ensure the best match
        "count": 1  # Limit to 1 result for clarity
    }

    try:
        # Search for the model
        logging.info(f"Searching Sketchfab for: {prompt}")
        search_resp = requests.get(search_url, headers=headers, params=params)
        search_resp.raise_for_status()
        result = search_resp.json()
        logging.info(f"Search response: {result}")

        if result.get("results"):
            model_info = result["results"][0]
            uid = model_info["uid"]
            model_name = sanitize_filename(prompt)
            logging.info(f"Model name determined: {model_name} (UID: {uid})")

            # Get download URL
            dl_url = f"https://api.sketchfab.com/v3/models/{uid}/download"
            dl_resp = requests.get(dl_url, headers=headers).json()
            logging.info(f"Download response: {dl_resp}")

            if "gltf" not in dl_resp:
                logging.error("No GLTF download link available.")
                return None
            model_url = dl_resp["gltf"]["url"]

            # Save zip with model name
            zip_path = os.path.join(CONTENT_DIR, f"{model_name}.zip")
            logging.info(f"Saving zip as: {zip_path}")
            with open(zip_path, "wb") as f:
                f.write(requests.get(model_url).content)

            # Extract to a folder named after the model
            model_extract_dir = os.path.join(extract_dir, model_name)
            os.makedirs(model_extract_dir, exist_ok=True)
            logging.info(f"Extracting to: {model_extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_extract_dir)

            # Find the model file
            for root, _, files in os.walk(model_extract_dir):
                for file in files:
                    if file.endswith((".glb", ".gltf", ".fbx")):
                        model_file_path = os.path.join(root, file)
                        logging.info(f"Model file found: {model_file_path}")
                        return model_file_path
        logging.warning(f"No results found for prompt: {prompt}")
        return None
    except Exception as e:
        logging.error(f"❌ Error downloading model: {str(e)}")
        return None

@cl.on_chat_start
async def start():
    await cl.Message(content="✅ Blender 4.3 Animation Script Generator ready! 🎨 Describe the animation you want (e.g., 'a car rotating').").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_prompt = message.content
    object_name = user_prompt.split()[0].lower()

    # Check for local model
    model_path = find_local_model(object_name)
    if not model_path:
        await cl.Message(content=f"🔍 No local model found for '{object_name}'. Downloading from Sketchfab...").send()
        model_path = search_and_download_model(object_name)
        if not model_path:
            await cl.Message(content=f"❌ No model found for '{object_name}' on Sketchfab.").send()
            return

    try:
        # Initialize Azure OpenAI client
        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )

        # System prompt for generating Blender animation scripts
        system_prompt = f"""
You are a Blender 4.3 Python scripting expert. Generate valid Python scripts for Blender 4.3 that create 3D animations based on user input. The scripts must:
- Be compatible with Blender 4.3 (Python 3.11).
- Use the provided model file at: {model_path}.
- Import the model using bpy.ops.import_scene.gltf() for .glb/.gltf files or bpy.ops.import_scene.fbx() for .fbx files.
- Create a 3D animation with keyframes for object movement, rotation, or scaling as described by the user.
- Include a camera that follows or focuses on the animated objects (e.g., using a 'TRACK_TO' constraint or dynamic positioning).
- Include at least one light source (e.g., Point, Sun, or Area light) with appropriate settings (e.g., energy=1000).
- Set the animation frame range (e.g., 1 to 100) with a default FPS of 24.
- Use the bpy module and follow Blender's Python API conventions.
- Include clear comments explaining the code.
- Ensure the script is ready to run in Blender's Text Editor with no external dependencies.
- Do NOT include rendering setup (e.g., resolution, output format, or render commands).
- If the user input is vague, make reasonable assumptions and document them in comments.
Output only the Python script inside a code block (```python\n<script>\n```).
"""

        completion = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate Blender 4.3 Python script for: {user_prompt}"}
            ],
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9
        )

        assistant_reply = completion.choices[0].message.content

        # Ensure the reply is formatted as a code block
        if not assistant_reply.startswith("```python"):
            assistant_reply = f"```python\n{assistant_reply.strip()}\n```"

        await cl.Message(content=assistant_reply ).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
