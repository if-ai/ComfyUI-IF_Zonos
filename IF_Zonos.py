# IF_Zonos.py
import os
import sys
import torch
import torchaudio
import folder_paths
from .zonos.model import Zonos
from .zonos.conditioning import make_cond_dict, supported_language_codes

class IF_ZonosTTS:
    """
    ComfyUI node for Zonos text-to-speech generation with emotion control.
    """
    def __init__(self):
        # Set required environment variables based on platform
        if os.name == 'nt':  # Windows
            os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
            os.environ["CUDA_HOME"] = os.environ.get("CUDA_PATH", "")
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
        elif sys.platform == 'darwin':  # macOS
            os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
            os.environ["CUDA_HOME"] = "/usr/local/cuda"  # Common CUDA path on macOS
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/local/lib/libespeak-ng.dylib"
        else:  # Linux and others
            os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
            os.environ["CUDA_HOME"] = "/usr/local/cuda"  # Common CUDA path on Linux
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/libespeak-ng.so.1"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_type = None
        # Disable torch inductor to avoid C++ compilation issues
        torch._dynamo.config.suppress_errors = True

    @classmethod
    def INPUT_TYPES(s):
        # Get path to silence file relative to the custom node's directory
        silence_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "silence_100ms.wav")
               
        return {
            "required": {
                "model_type": (["Transformer", "Hybrid"], {"default": "Transformer"}),
                "text": ("STRING", {"multiline": True, "default": "Welcome to Zonos text to speech!"}),
                "language": (supported_language_codes, {"default": "en-us"}),
                "happiness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sadness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "disgust": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fear": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "surprise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "anger": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "other": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "neutral": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "min_p": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 420, "min": 0, "max": 0xfffffffff}),
            },
            "optional": {
                "speaker_audio": ("AUDIO",),
                "prefix_audio": ("AUDIO", {
                    "default": silence_path if os.path.exists(silence_path) else None
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "ImpactFrames💥🎞️/Audio"

    def load_model_if_needed(self, model_type):
        if self.model is None or self.model_type != model_type:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            print(f"Loading {model_type} model...")
            if model_type == "Transformer":
                self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
            else:
                self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=self.device)
            
            self.model.to(self.device)
            self.model.bfloat16()
            self.model.eval()
            self.model_type = model_type

    def preprocess_audio(self, waveform, sample_rate, target_sr):
        """Safely process audio input."""
        try:
            waveform = waveform.squeeze()  # Remove any extra dimensions
            
            # Handle multi-channel audio
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)  # Convert to mono
            
            # Add batch dimension if needed
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
                
            return waveform.to(self.device)
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def generate_speech(self, model_type, text, language, happiness, sadness, 
                    disgust, fear, surprise, anger, other, neutral,
                    cfg_scale, min_p, seed, speaker_audio=None, prefix_audio=None):
        try:
            self.load_model_if_needed(model_type)
            
            # Set random seed  
            torch.manual_seed(seed)
            
            # Process speaker audio if provided
            speaker_embedding = None
            if speaker_audio is not None:
                waveform = self.preprocess_audio(
                    speaker_audio["waveform"],
                    speaker_audio["sample_rate"], 
                    16000
                )
                if waveform is not None:
                    speaker_embedding = self.model.make_speaker_embedding(
                        waveform,
                        16000
                    )
                    speaker_embedding = speaker_embedding.to(self.device, dtype=torch.bfloat16)

            # Process prefix audio if provided
            audio_prefix_codes = None 
            if prefix_audio is not None:
                # Use the same approach as in the gradio interface:
                waveform = prefix_audio["waveform"]
                # Ensure mono channel with channel dimension preserved
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:
                    waveform = waveform.unsqueeze(0)
                sr_prefix = prefix_audio["sample_rate"]
                try:
                    # Use autoencoder's preprocess for proper handling
                    waveform = self.model.autoencoder.preprocess(waveform, sr_prefix)
                    waveform = waveform.to(self.device, dtype=torch.float32)
                    # Autoencoder.encode expects a batch dimension; add one if missing.
                    if waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)
                    audio_prefix_codes = self.model.autoencoder.encode(waveform)
                except Exception as e:
                    print(f"Warning: Could not process prefix audio: {e}")
                    audio_prefix_codes = None

            # Create emotion tensor
            emotion_tensor = torch.tensor(
                [[happiness, sadness, disgust, fear, surprise, anger, other, neutral]],
                device=self.device
            )
            
            # Create condition dictionary
            cond_dict = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                device=self.device
            )

            # Generate audio (using autocast on CUDA for improved performance/quality)
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        conditioning = self.model.prepare_conditioning(cond_dict)
                        codes = self.model.generate(
                            prefix_conditioning=conditioning,
                            audio_prefix_codes=audio_prefix_codes,
                            max_new_tokens=86 * 30,
                            cfg_scale=cfg_scale,
                            batch_size=1,
                            sampling_params=dict(min_p=min_p),
                            disable_torch_compile=True
                        )
                        # Decode audio after generation
                        wav_out = self.model.autoencoder.decode(codes)
                else:
                    conditioning = self.model.prepare_conditioning(cond_dict)
                    codes = self.model.generate(
                        prefix_conditioning=conditioning,
                        audio_prefix_codes=audio_prefix_codes,
                        max_new_tokens=86 * 30,
                        cfg_scale=cfg_scale,
                        batch_size=1,
                        sampling_params=dict(min_p=min_p),
                        disable_torch_compile=True
                    )
                    wav_out = self.model.autoencoder.decode(codes)
                    
                # Ensure proper shape for the output audio
                wav_out = wav_out.cpu()
                if wav_out.dim() == 1:
                    wav_out = wav_out.unsqueeze(0)  # Add channels dimension
                elif wav_out.dim() == 2:
                    if wav_out.size(0) > 1:
                        wav_out = wav_out[0:1, :]  # Take only the first channel to preserve cadence
                elif wav_out.dim() == 3:
                    wav_out = wav_out.squeeze(0)  # Remove batch dimension if present
                    
                # Return audio in ComfyUI format
                return ({"waveform": wav_out.unsqueeze(0), 
                        "sample_rate": self.model.autoencoder.sampling_rate},)

        except Exception as e:
            print(f"Error in generate_speech: {str(e)}")
            raise e

    def time_shift(self, audio, speed):
        import torch.nn.functional as F
        rate = audio['sample_rate']
        waveform = audio['waveform']
        
        # Correct speed factor:
        # If speed > 1, audio should be slower (longer duration),
        # so we multiply the original length by speed.
        old_length = waveform.shape[-1]
        new_length = int(old_length * speed)
        
        # Resample audio using linear interpolation
        new_waveform = F.interpolate(
            waveform.unsqueeze(1),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)
    
        return {"waveform": new_waveform, "sample_rate": rate}

NODE_CLASS_MAPPINGS = {
    "IF_ZonosTTS": IF_ZonosTTS  
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_ZonosTTS": "Zonos Text to Speech 🎤"
}
