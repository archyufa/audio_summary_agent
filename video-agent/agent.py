import datetime
from zoneinfo import ZoneInfo
import uuid
import base64
import requests
import io
import time
import json
import os
import re
import urllib.parse
import asyncio
import subprocess
import tempfile
import shutil

# Google Cloud Imports
import google.auth
import google.auth.transport.requests
from google.cloud import storage
from google.cloud import texttospeech
from google.cloud import speech
from mutagen.mp3 import MP3

# Audio Processing
from pydub import AudioSegment

# Vertex AI Imports
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview.vision_models import VideoGenerationModel # <--- For Veo

# ADK Imports
from google.adk.agents import Agent, LlmAgent
from google.adk.tools import AgentTool, ToolContext, load_artifacts, google_search 
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.artifacts import InMemoryArtifactService 
from google.genai import types 

# Configuration
GOOGLE_GENAI_USE_VERTEXAI = True
GOOGLE_CLOUD_PROJECT = "all-in-demo"
GOOGLE_CLOUD_LOCATION = "global"

# --- TOOLS (State Managed via Artifacts) ---

async def save_timeline(html_code: str, tool_context: ToolContext) -> str:
    """Saves the HTML timeline code as an artifact."""
    try:
        html_bytes = html_code.encode("utf-8")
        artifact_part = types.Part.from_bytes(data=html_bytes, mime_type="text/html")
        version = await tool_context.save_artifact(filename="timeline.html", artifact=artifact_part)
        return f"Timeline saved successfully (Version {version})."
    except Exception as e:
        return f"Error saving timeline: {e}"

async def get_timeline(tool_context: ToolContext) -> str:
    """Retrieves the current HTML timeline code."""
    try:
        artifact = await tool_context.load_artifact(filename="timeline.html")
        if not artifact or not artifact.inline_data:
            return "Error: No timeline found. Please run the Timeline Generator first."
        return artifact.inline_data.data.decode("utf-8")
    except Exception as e:
        return f"Error loading timeline: {e}"

async def replace_scene_code(scene_number: int, new_function_code: str, tool_context: ToolContext) -> str:
    """Integrates a specific scene's JS code into the master HTML artifact using Robust Regex."""
    print(f"🔧 Integrating code for Scene {scene_number}...")

    current_html_code = await get_timeline(tool_context=tool_context)
    if current_html_code.startswith("Error"):
        return current_html_code

    clean_code = new_function_code.replace("```javascript", "").replace("```", "").strip()
    
    start_pattern = r"//\s*SCENE_" + str(scene_number) + r"_START"
    end_pattern = r"//\s*SCENE_" + str(scene_number) + r"_END"
    
    pattern = re.compile(f"({start_pattern})(.*?)({end_pattern})", re.DOTALL)
    
    match = pattern.search(current_html_code)
    if not match:
        return f"Error: Markers for Scene {scene_number} not found. Ensure Timeline Generator created `// SCENE_{scene_number}_START`."

    replacement = f"// SCENE_{scene_number}_START\n{clean_code}\n// SCENE_{scene_number}_END"
    updated_html = pattern.sub(replacement, current_html_code)

    return await save_timeline(html_code=updated_html, tool_context=tool_context)

# --- STANDARD TOOLS (GCS / Audio / Video / Veo) ---

def generate_veo_background(prompt: str) -> str:
    """
    Generates a video background using Vertex AI Veo (Image 3/Video).
    Returns the GCS URL of the generated MP4.
    """
    bucket_name = "mad-apple-photos"
    storage_client = storage.Client()
    
    print(f"🎨 Generating Veo Background with prompt: '{prompt}'...")

    try:
        # Load the Veo model (adjust model name if 'veo-001' changes in your region)
        model = VideoGenerationModel.from_pretrained("veo-001")
        
        # Generate video (Veo typically takes 10-60s)
        response = model.generate_video(
            prompt=prompt + ", seamless loop, 4k, abstract background, slow motion, high quality",
            aspect_ratio="9:16",
            person_generation="dont_allow" # Strictly background
        )
        
        # Save to GCS
        # Veo responses usually provide a uri or bytes. We handle the bytes.
        video_bytes = response.video_bytes
        
        blob_name = f"veo_backgrounds/bg_{uuid.uuid4()}.mp4"
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(video_bytes, content_type="video/mp4")
        
        print(f"✅ Veo Background Generated: {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print(f"❌ Veo Generation Failed: {e}")
        return "NONE" # Return NONE so the agent knows to skip the background tag

def list_audio_files() -> str:
    """Lists all MP3 files available in the 'radio-canada-audio-test' bucket."""
    bucket_name = "radio-canada-audio-test"
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)
        mp3_files = [blob.name for blob in blobs if blob.name.lower().endswith(".mp3")]
        
        if not mp3_files:
            return "No MP3 files found in the bucket."
        
        # Format list for the LLM
        return "Available Audio Files:\n" + "\n".join([f"- https://storage.googleapis.com/{bucket_name}/{f}" for f in mp3_files])
    except Exception as e:
        return f"Error listing files: {str(e)}"

def cut_audio_snippet(source_url: str, start_time: float, end_time: float) -> str:
    """Cuts a segment from a GCS audio file."""
    bucket_name = "mad-apple-photos"
    storage_client = storage.Client()
    
    if source_url.startswith("https://"): parts = source_url.replace("https://storage.googleapis.com/", "").split("/", 1)
    elif source_url.startswith("gs://"): parts = source_url.replace("gs://", "").split("/", 1)
    else: return "Error: Invalid URL"
    
    bucket_src, blob_name = parts[0], urllib.parse.unquote(parts[1])

    try:
        blob = storage_client.bucket(bucket_src).blob(blob_name)
        audio = AudioSegment.from_file(io.BytesIO(blob.download_as_bytes()))
        snippet = audio[start_time*1000:end_time*1000]
        
        out_buffer = io.BytesIO()
        snippet.export(out_buffer, format="mp3")
        out_buffer.seek(0)
        
        target_blob = storage_client.bucket(bucket_name).blob(f"snippets/snippet_{uuid.uuid4()}.mp3")
        target_blob.upload_from_file(out_buffer, content_type="audio/mpeg")
        return target_blob.public_url
    except Exception as e: return f"Error: {e}"

def transcribe_from_gcs_url(gcs_url: str) -> str:
    """Transcribes audio from GCS with caching using Gemini 2.5 Flash."""
    storage_client = storage.Client()
    
    if gcs_url.startswith("https://"): 
        clean_url = gcs_url.replace("https://storage.googleapis.com/", "gs://")
        parts = gcs_url.replace("https://storage.googleapis.com/", "").split("/", 1)
    elif gcs_url.startswith("gs://"): 
        clean_url = gcs_url
        parts = gcs_url.replace("gs://", "").split("/", 1)
    else: 
        return json.dumps({"error": "Invalid URL format"})

    bucket_name, blob_name = parts[0], urllib.parse.unquote(parts[1])
    
    cache_blob = storage_client.bucket(bucket_name).blob(blob_name.rsplit('.', 1)[0] + ".txt")
    if cache_blob.exists():
        print(f"✅ Found cached transcript for {blob_name}")
        return cache_blob.download_as_text()

    print(f"🎙️ Transcribing {clean_url} using Gemini 2.5 Flash...")

    try:
        model = GenerativeModel("gemini-2.5-flash") 
        audio_part = Part.from_uri(mime_type="audio/mpeg", uri=clean_url)
        response = model.generate_content(
            [audio_part, "Transcribe this audio file verbatim. Do not add any commentary or introductory text. the transcription should have the name of the person talking when you can,like a movie script."],
            generation_config={"temperature": 0.0}
        )
        full_transcript = response.text
        output = json.dumps({"transcript": full_transcript})
        cache_blob.upload_from_string(output)
        return output
    except Exception as e:
        return json.dumps({"error": f"Transcription failed: {str(e)}"})

def generate_tts_and_upload(script_text: str, language: str = "en-US") -> str:
    """Generates Gemini 2.5 TTS audio with a Professional Broadcast prompt."""
    bucket_name = "mad-apple-photos"
    lang_raw = language.lower().strip()
    
    if "fr" in lang_raw or "french" in lang_raw:
        lang_code = "fr-CA"
        voice_name = "Charon"
        system_prompt = "Tu es un animateur radio énergique. Parle vite et bien. Ton: captivant."
    else:
        lang_code = "en-US"
        voice_name = "Fenrir"
        system_prompt = "You are a seasoned public broadcaster (like BBC/NPR). Your tone is credible, articulate, warm, and informative. Speak with clarity."

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=script_text, prompt=system_prompt),
        voice=texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name, model_name="gemini-2.5-pro-tts"),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    )
    blob = storage.Client().bucket(bucket_name).blob(f"audio/tts_{uuid.uuid4()}.mp3")
    blob.upload_from_string(response.audio_content, content_type="audio/mpeg")
    duration = MP3(io.BytesIO(response.audio_content)).info.length
    return json.dumps({"audio_url": blob.public_url, "duration": duration, "language": lang_code})

def stitch_audio_clips(audio_urls: list[str]) -> str:
    return "https://storage.googleapis.com/mad-apple-photos/audio/master_audio_EXAMPLE.mp3" 

async def write_html_to_gcs(tool_context: ToolContext) -> str:
    """Fetches the 'timeline.html' artifact and writes it to GCS for previewing."""
    try:
        artifact = await tool_context.load_artifact(filename="timeline.html")
        if not artifact or not artifact.inline_data:
            return "Error: No timeline artifact found to publish."
        
        html_string = artifact.inline_data.data.decode("utf-8")
        
        blob = storage.Client().bucket("mad-apple-photos").blob(f"{uuid.uuid4()}.html")
        blob.upload_from_string(html_string, content_type='text/html')
        return f"Preview Link: {blob.public_url}"
    except Exception as e:
        return f"Error publishing preview: {e}"

def export_to_video(html_gcs_url: str) -> str:
    """
    Exports a GCS HTML URL to an MP4 video.
    OPTIMIZED: Forces 30FPS and ensures timeout safety.
    """
    bucket_name = "mad-apple-photos"
    storage_client = storage.Client()

    print(f"🎬 Starting Video Export for: {html_gcs_url}")

    if not shutil.which("npx"):
        return "Error: 'npx' executable not found in PATH."

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_mp4_path = os.path.join(temp_dir, "output.mp4")

            # 1. Run Node Export Tool
            cmd = [
                "npx", "gsap-video-export",
                "-i", html_gcs_url,
                "-o", local_mp4_path,
                "-W", "1080", 
                "-H", "1920",
                "-f", "30",       # Force 30 FPS
                "-S", "body",
                "-D", "30"        # Duration
            ]
            
            print(f"⚙️ Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=temp_dir,
                timeout=600 # 10 Minute Timeout
            )

            # 2. Check and Upload
            if os.path.exists(local_mp4_path) and os.path.getsize(local_mp4_path) > 1000:
                target_blob_name = f"videos/final_{uuid.uuid4()}.mp4"
                target_blob = storage_client.bucket(bucket_name).blob(target_blob_name)
                target_blob.upload_from_filename(local_mp4_path, content_type="video/mp4")
                
                return f"✅ Video Export Complete: {target_blob.public_url}"
            else:
                error_msg = (
                    f"Error: Output MP4 file was not created.\n"
                    f"--- COMMAND OUTPUT ---\n{result.stdout}\n"
                    f"--- COMMAND ERROR ---\n{result.stderr}\n"
                )
                print(error_msg)
                return error_msg

    except subprocess.TimeoutExpired:
        return "Error: Video encoding timed out after 10 minutes."
    except Exception as e:
        return f"System Error during export: {str(e)}"

# --- AGENTS ---

researcher = LlmAgent(
    name="researcher",
    model="gemini-3-pro-preview", 
    description="Fact & Name Verifier.",
    instruction=(
        """You are the **Lead Researcher**.
        **INPUT:** Transcript Text.
        **GOAL:** Verify names, dates, and key claims via Google Search.
        **OUTPUT:** A "Research Brief" listing verified facts.
        """
    ),
    tools=[google_search]
)

creative_director = LlmAgent(
    name="creative_director",
    model="gemini-3-pro-preview",
    description="Creative Director. Ideates and writes scripts.",
    instruction=(
        """You are the **Creative Director** for a vertical video series.
        
        **INPUT:** Transcript Text + Research Brief.
        
        **TASK 1: IDEATION (If asked for ideas)**
        - Generate 5 distinct angles/hooks.
        
        **TASK 2: STORYBOARDING (If asked for storyboard)**
        - Write a full script for Kinetic Typography.
        - **NEW: BACKGROUND VIDEO.** Decide if this video needs a generated abstract background.
        
        **MANDATORY OUTPUT FORMAT:**
        
        **GLOBAL THEME:**
        BACKGROUND_PROMPT: "[Description of abstract 3D background, e.g., 'Dark neon geometric shapes floating in void, 4k, loop']" (Or 'NONE' if solid black is better).
        
        **SCENE 1 SCRIPT:** "[Text]"
        **STYLE/FEEL:** "[Adjectives]"
        
        **SCENE 2 SCRIPT:** "[Text]"
        **STYLE/FEEL:** "[Adjectives]"
        """
    ),
    tools=[]
)

timeline_generator = LlmAgent(
    name="timeline_generator",
    model="gemini-3-pro-preview",
    description="GSAP Architect. Generates the Master Timeline Skeleton.",
    instruction=(
        """You are the **Master GSAP Architect**.
        
        **INPUT:** You will receive a `background_video_url` (optional) in the prompt.
        
        **GOAL:** Create `index.html` (1080x1920).
        
        **BACKGROUND LOGIC:**
        - If `background_video_url` is provided (and not 'NONE'), inject a `<video>` tag immediately inside `#stage`.
        - CSS for Video: `position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 0; opacity: 0.6;`
        - **IMPORTANT:** Ensure scene containers sit ON TOP of this video (z-index > 1).
        
        **STRUCTURE:**
        1. HTML Skeleton.
        2. `#stage` div.
        3. **(Optional) Background Video.**
        4. GSAP Script (`master` timeline).
        5. Global Functions: `scene1()`, `scene2()`... defined as placeholders.
        
        **OUTPUT:** Valid HTML string.
        """
    ),
    tools=[]
)

scene_creator = LlmAgent(
    name="scene_creator",
    model="gemini-3-pro-preview",
    description="GSAP Animator.",
    instruction=(
        """You are a **GSAP Animator**.
        **MISSION:** Write the JS function for **ONE specific scene**.
        **CONSTRAINT:** Transparency.
        - Do NOT set a background color on your container (`cont.style.backgroundColor`). 
        - Let the global video background show through.
        - Use White/Bright text for contrast.
        
        **OUTPUT:** ONLY the JS function `function sceneX() { ... }`.
        """
    ),
    tools=[]
)

# --- ROOT AGENT ---

root_agent = Agent(
    name="root_agent",
    model="gemini-3-pro-preview",
    description="Orchestrator.",
    instruction=("""
    You are the **Animation Orchestrator**.
    
    **Workflow:**

    **PHASE 1: SETUP**
    1.  **Start:** Call `list_audio_files`, ask user to pick one.
    2.  **Transcribe:** Call `transcribe_from_gcs_url`.
    3.  **Ideate:** Call `creative_director` for 5 ideas. Ask user to pick.
    4.  **Research:** Call `researcher`.

    **PHASE 2: PLAN & ASSETS**
    5.  **Storyboard:** Call `creative_director` (Pass Transcript + Research).
        - **CRITICAL:** Look for `BACKGROUND_PROMPT:` in the output.
    6.  **Background (NEW):** - If `BACKGROUND_PROMPT` is found and not "NONE":
        - Call `generate_veo_background(prompt)`.
        - **Save the returned URL.**
    7.  **Audio:** Call `generate_tts_and_upload`.
    
    **PHASE 3: SKELETON**
    8.  **Skeleton:** Call `timeline_generator`. 
        - **PASS the `background_video_url` (if generated) in the prompt.**
        - Save via `save_timeline`.
    9.  **Preview:** Call `write_html_to_gcs`.

    **PHASE 4: SCENES**
    *Loop scenes 1 to N:*
    10. **Generate Scene:** Call `scene_creator`.
    11. **Integrate:** Call `replace_scene_code`.
    12. **Preview:** Call `write_html_to_gcs`.

    **PHASE 5: FINAL**
    13. **Export:** Call `export_to_video`.
    """),
    tools=[
        list_audio_files,
        transcribe_from_gcs_url,
        generate_tts_and_upload,
        generate_veo_background, # <--- NEW TOOL
        stitch_audio_clips,
        write_html_to_gcs,
        export_to_video,
        save_timeline,       
        get_timeline,       
        replace_scene_code,
        google_search,
        AgentTool(creative_director),
        AgentTool(timeline_generator),
        AgentTool(scene_creator),
        AgentTool(researcher),
        load_artifacts
    ]
)

# --- APP SETUP ---
local_app = App(
    name="AnimationOrchestrator",
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)

if __name__ == "__main__":
    import vertexai
    from vertexai import agent_engines
    from vertexai.preview import reasoning_engines

    # Initialize Vertex AI
    vertexai.init(
        project="all-in-demo",
        location="us-central1",
        staging_bucket="gs://staging_agent_ttest",
    )
    
    # Create the Vertex AI Reasoning Engine wrapper
    app = reasoning_engines.AdkApp(
        agent=local_app, 
        enable_tracing=True
    )
    
    # Deploy the agent
    remote_app = agent_engines.create(
        agent_engine=app,
        requirements=[
            "google-adk==1.15.1",
            "google-cloud-aiplatform[agent_engines]",
            "google-cloud-discoveryengine",
            "google-cloud-storage",
            "google-cloud-texttospeech>=2.29.0", 
            "google-cloud-speech",
            "mutagen",
            "pydub",
            "vertexai"
        ]
    )