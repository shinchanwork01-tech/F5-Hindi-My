import os
import shutil
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
import torch
from cached_path import cached_path

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

# =================================================================
# 1. PRE-DEFINED VOICES
# =================================================================
# Ensure these .wav files exist in /content/voices/

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
    os.makedirs(SAVE_PATH, exist_ok=True)

# =================================================================
# 2. MODEL LOADING
# =================================================================
print("--- Loading Hindi Model Components ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder()

def load_f5tts_small():
    ckpt_path = str(cached_path("hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors"))
    vocab_path = str(cached_path("hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt"))
    model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

ema_model = load_f5tts_small()
print(f"--- Model Loaded on {device} ---")

# =================================================================
# 3. LOGIC FUNCTIONS
# =================================================================

def synthesize_hindi(voice_name, gen_text):
    if not voice_name or not gen_text:
        return None, "⚠️ Please select a voice and enter text."
    
    voice_data = VOICES.get(voice_name)
    ref_audio_path = voice_data["ref_audio"]
    ref_text_val = voice_data["ref_text"]

    if not os.path.exists(ref_audio_path):
        return None, f"⚠️ Error: File not found: {ref_audio_path}"

    try:
        print(f"Processing: {voice_name}")
        # Pre-process reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_val)
        
        # Inference (Removed 'indic=True' to fix the TypeError)
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            cross_fade_duration=0.15,
            speed=1.0,
            nfe_step=32
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            return f.name, "✅ Synthesis Complete"
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"⚠️ Error: {str(e)}"

def save_to_folder(temp_audio_path):
    if not temp_audio_path:
        return "⚠️ No audio to save."
    
    try:
        existing_files = os.listdir(SAVE_PATH)
        count = 1
        while f"output_{count}.wav" in existing_files:
            count += 1
        
        final_filename = f"output_{count}.wav"
        destination = os.path.join(SAVE_PATH, final_filename)
        shutil.copy(temp_audio_path, destination)
        return f"✅ Saved as {final_filename}"
    except Exception as e:
        return f"⚠️ Save Error: {str(e)}"

# =================================================================
# 4. GRADIO UI
# =================================================================

with gr.Blocks(title="Hindi TTS Station") as app:
    gr.Markdown("# 🎙️ Hindi Voice Station")
    
    last_audio_temp = gr.State("")

    with gr.Column(variant="panel"):
        voice_choice = gr.Dropdown(
            choices=list(VOICES.keys()), 
            label="1. Select Voice", 
            value=list(VOICES.keys())[0] if VOICES else None
        )
        
        input_text = gr.Textbox(
            label="2. Enter Hindi Text", 
            placeholder="यहाँ अपना हिंदी टेक्स्ट लिखें...", 
            lines=5
        )
        
        with gr.Row():
            synth_btn = gr.Button("Synthesize", variant="primary")
            save_btn = gr.Button("Save to Folder")
        
        audio_out = gr.Audio(
            label="3. Preview Result", 
            type="filepath", 
            interactive=False
        )
        
        status_msg = gr.Textbox(label="Status", interactive=False)

    # Actions
    synth_btn.click(
        fn=synthesize_hindi,
        inputs=[voice_choice, input_text],
        outputs=[audio_out, status_msg]
    ).then(
        fn=lambda x: x,
        inputs=[audio_out],
        outputs=[last_audio_temp]
    )

    save_btn.click(
        fn=save_to_folder,
        inputs=[last_audio_temp],
        outputs=[status_msg]
    )

if __name__ == "__main__":
    app.launch(share=True)
