from diffusers import AudioLDM2Pipeline
import torch
import scipy
import random
repo_id = "cvssp/audioldm2-music"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype = torch.float16)
pipe = pipe.to("cuda")

negative_prompt = "Low quality."
prompt = "a composition with a sad mood, a slow rhythm, a low pitch, a soft intensity, and a mellow timbre."
for i in range(30):
  generator = torch.Generator("cuda").manual_seed(random.randint(0, 100000))
  audio = pipe(prompt, num_inference_steps = 250, negative_prompt = negative_prompt, audio_length_in_s = 15.0, generator = generator).audios[0]
  scipy.io.wavfile.write("generated/{}.mp3".format(str(i)), rate = 16000, data = audio)