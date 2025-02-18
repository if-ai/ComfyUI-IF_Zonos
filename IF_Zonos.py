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
        torch._dynamo.config.suppress_errors = True

    @classmethod
    def INPUT_TYPES(s):
        silence_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "silence_100ms.wav")
        return {
            "required": {
                "model_type": (["Transformer", "Hybrid"], {"default": "Transformer"}),
                "text": ("STRING", {"multiline": True, "default": "Welcome to Zonos! 🎤"}),
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
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 420, "min": 0, "max": 0xfffffffff}),
            },
            "optional": {
                "speaker_audio": ("AUDIO",),
                "prefix_audio": ("AUDIO", {"default": silence_path if os.path.exists(silence_path) else None}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "Audio"

    def load_model_if_needed(self, model_type):
        if self.model is None or self.model_type != model_type:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            print(f"Loading {model_type} model...")
            if model_type == "Transformer":
                self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
            elif model_type == "Hybrid":
                self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=self.device)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            self.model.to(self.device)
            self.model.bfloat16()
            self.model.eval()
            self.model_type = model_type

    def preprocess_audio(self, waveform, sample_rate, target_sr):
        try:
            waveform = waveform.squeeze()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.unsqueeze(0)
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            return waveform.to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def generate_speech(self, model_type, text, language, happiness, sadness,
                        disgust, fear, surprise, anger, other, neutral,
                        cfg_scale, speed, seed, speaker_audio=None, prefix_audio=None):
        try:
            self.load_model_if_needed(model_type)
            torch.manual_seed(seed)

            speaker_embedding = None
            if speaker_audio is not None:
                waveform = self.preprocess_audio(speaker_audio["waveform"], speaker_audio["sample_rate"], 16000)
                if waveform is not None:
                    speaker_embedding = self.model.make_speaker_embedding(waveform, 16000)
                    speaker_embedding = speaker_embedding.to(torch.bfloat16)

            audio_prefix_codes = None
            if prefix_audio is not None:
                waveform = prefix_audio["waveform"]
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:
                    waveform = waveform.unsqueeze(0)
                sr_prefix = prefix_audio["sample_rate"]
                try:
                    waveform = self.model.autoencoder.preprocess(waveform, sr_prefix)
                    waveform = waveform.to(self.device, dtype=torch.bfloat16)
                    waveform = waveform.unsqueeze(0)
                    audio_prefix_codes = self.model.autoencoder.encode(waveform)
                except Exception as e:
                    print(f"Warning: Could not process prefix audio: {e}")
                    audio_prefix_codes = None

            min_speaking_rate = 5.0
            max_speaking_rate = 40.0
            speaking_rate = max(min_speaking_rate, min(15.0 * speed, max_speaking_rate))

            emotion_tensor = torch.tensor(
                [[happiness, sadness, disgust, fear, surprise, anger, other, neutral]],
                dtype=torch.bfloat16,
                device=self.device
            )

            cond_dict = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                speaking_rate=speaking_rate,
                device=self.device
            )

            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.amp.autocast("cuda"):
                        conditioning = self.model.prepare_conditioning(cond_dict)
                        codes = self.model.generate(
                            prefix_conditioning=conditioning,
                            audio_prefix_codes=audio_prefix_codes,
                            max_new_tokens=86 * 30,
                            cfg_scale=cfg_scale,
                            batch_size=1,
                            sampling_params=dict(),
                            disable_torch_compile=True
                        )
                        wav_out = self.model.autoencoder.decode(codes)
                else:
                    conditioning = self.model.prepare_conditioning(cond_dict)
                    codes = self.model.generate(
                        prefix_conditioning=conditioning,
                        audio_prefix_codes=audio_prefix_codes,
                        max_new_tokens=86 * 30,
                        cfg_scale=cfg_scale,
                        batch_size=1,
                        sampling_params=dict(),
                        disable_torch_compile=True
                    )
                    wav_out = self.model.autoencoder.decode(codes)

                wav_out = wav_out.cpu().float()
                if wav_out.dim() == 1:
                    wav_out = wav_out.unsqueeze(0)
                elif wav_out.dim() == 2 and wav_out.size(0) > 1:
                    wav_out = wav_out[0:1, :]

                return ({"waveform": wav_out.unsqueeze(0), "sample_rate": self.model.autoencoder.sampling_rate},)

        except Exception as e:
            print(f"Error in generate_speech: {str(e)}")
            raise e

NODE_CLASS_MAPPINGS = {"IF_ZonosTTS": IF_ZonosTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_ZonosTTS": "Zonos Text to Speech 🎤" }
