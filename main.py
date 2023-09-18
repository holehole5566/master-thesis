import torchaudio
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
torch.cuda.empty_cache()
model = MusicGen.get_pretrained('small')
model.set_generation_params(duration = 30)  # generate 30 seconds.
prompt = "Create a serene and calming musical composition with very low intensity, soft timbre, low pitch, and an extremely slow, tranquil rhythm to convey a profound sense of relaxation and peacefulness in the music."
descriptions = []
descriptions.append(prompt)
for i in range(100):
    wav = model.generate(descriptions) # generate 100 pieces of music.
    for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{i}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)