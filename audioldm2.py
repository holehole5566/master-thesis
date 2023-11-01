from diffusers import AudioLDM2Pipeline
import torch
import scipy
repo_id = "cvssp/audioldm2-music"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype = torch.float16)
pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
negative_prompt = "Low quality."
prompt = "A funky bass guitar grooving in sync with drums."
for i in range(3):
  audio = pipe(prompt, num_inference_steps = 200, negative_prompt = negative_prompt, audio_length_in_s = 20.0, generator = generator).audios[0]
  scipy.io.wavfile.write(str(i) + ".wav", rate = 16000, data = audio)