import os
import shutil
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

# =================================================================
# 1. PRE-DEFINED VOICES (Change paths and text here)
# =================================================================
VOICES = {
    "Short": {
        "ref_audio": "/content/F5-Hindi-My/audio2.wav",
        "ref_text": "वेन यू गेट स्टिचेस, ए डॉक्टर थ्रेड्स ए नीडल थ्रू योर वुन्ड एंड पुल्स इट टाइट टू क्लोज द स्किन।"
    },
    "Kurt": {
        "ref_audio": "/content/F5-Hindi-My/audio1.wav",
        "ref_text": "परमानेंट टैटू बनवाना पिछले कुछ सालों में एक ट्रेंड बन गया है।"
    }
}

SAVE_PATH = "/content/audio/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# =================================================================
# 2. MODEL LOADING (HINDI SMALL ONLY)
# =================================================================
print("Starting Hindi-Only TTS Station...")
vocoder = load_vocoder()

def load_f5tts_small():
    ckpt_path = str(cached_path("hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors"))
    vocab_path = str(cached_path("hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt"))
    # Small model configuration
    model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

# Load only the Hindi model to save VRAM
ema_model = load_f5tts_small()

# =================================================================
# 3. LOGIC FUNCTIONS
# =================================================================

def synthesize_hindi(voice_name, gen_text):
    if not voice_name or not gen_text:
        return None, "Error: Select a voice and enter text."
    
    ref_audio_path = VOICES[voice_name]["ref_audio"]
    ref_text_val = VOICES[voice_name]["ref_text"]
    
    # Pre-process reference audio
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_val)
    
    # Run Inference
    # Defaults: speed=1.0, nfe=32, cross_fade=0.15, indic=True
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=0.15,
        speed=1.0,
        nfe_step=32,
        indic=True,
        show_info=print
    )
    
    # Save to a temporary file for the Gradio player
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, final_wave, final_sample_rate)
        return f.name

def save_to_folder(temp_audio_path):
    if not temp_audio_path or not os.path.exists(temp_audio_path):
        return "❌ No audio generated yet."
    
    # Option A: Sequential Naming (output_1.wav, output_2.wav...)
    existing_files = os.listdir(SAVE_PATH)
    count = 1
    while f"output_{count}.wav" in existing_files:
        count += 1
    
    final_filename = f"output_{count}.wav"
    destination = os.path.join(SAVE_PATH, final_filename)
    
    shutil.copy(temp_audio_path, destination)
    return f"✅ Saved as {final_filename} in {SAVE_PATH}"

# =================================================================
# 4. GRADIO UI LAYOUT (CLEAN VERTICAL STACK)
# =================================================================

with gr.Blocks(title="Hindi TTS Station") as app:
    gr.Markdown("# 🎙️ Hindi Voice Station")
    gr.Markdown("Select a voice, enter Hindi text, and click Synthesize.")
    
    # Hidden state to track the temporary audio path
    last_audio_temp = gr.State("")

    with gr.Column(variant="panel"):
        # 1. Inputs
        voice_choice = gr.Dropdown(
            choices=list(VOICES.keys()), 
            label="Select Voice", 
            value=list(VOICES.keys())[0] if VOICES else None
        )
        
        input_text = gr.Textbox(
            label="Hindi Text Input", 
            placeholder="यहाँ अपना हिंदी टेक्स्ट लिखें...", 
            lines=6
        )
        
        # 2. Buttons
        with gr.Row():
            synth_btn = gr.Button("Synthesize", variant="primary")
            save_btn = gr.Button("Save to Folder")
        
        # 3. Output Player (interactive=False removes the 'Upload' UI)
        audio_out = gr.Audio(
            label="Synthesized Audio Preview", 
            type="filepath", 
            interactive=False
        )
        
        # 4. Status Message
        status_msg = gr.Markdown("Ready.")

    # --- Event Handling ---
    
    # When Synthesize is clicked
    synth_btn.click(
        fn=synthesize_hindi,
        inputs=[voice_choice, input_text],
        outputs=[audio_out]
    ).then(
        fn=lambda x: x,
        inputs=[audio_out],
        outputs=[last_audio_temp]
    )

    # When Save is clicked
    save_btn.click(
        fn=save_to_folder,
        inputs=[last_audio_temp],
        outputs=[status_msg]
    )

if __name__ == "__main__":
    # Launch Gradio with the --share functionality as per your original command

    app.launch(share=True)
