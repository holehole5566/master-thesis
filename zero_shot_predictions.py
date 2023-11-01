"""
This is an example using CLAP for zero-shot inference.
"""
from msclap import CLAP
import random
import torch.nn.functional as F

# Define classes for zero-shot
# Should be in lower case and can be more than one word
classes = ['high arousal and positive valence','high arousal and negative valence','low arousal and negative valence', 'low arousal and positive valence']
ground_truth = ['coughing']
# Add prompt
prompt = 'this is a music with '
class_prompts = []
audio_files = []
for desc in classes:
    prompt += desc
    class_prompts.append(prompt)
    
#Load audio files
for i in range(1):
    i = random.randint(1, 225)
    #music = 'music (' + str(i) + ').mp3'
    music = '0.wav'
    if music in audio_files:
        i -= 1
        continue
    audio_files.append(music)

# Load and initialize CLAP
# Setting use_cuda = True will load the model on a GPU using CUDA
clap_model = CLAP(version = '2023', use_cuda=False)

# compute text embeddings from natural text
text_embeddings = clap_model.get_text_embeddings(class_prompts)

# compute the audio embeddings from an audio file
audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)

# compute the similarity between audio_embeddings and text_embeddings
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(4)

# Print the results
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

"""
The output (the exact numbers may vary):

Ground Truth: coughing
Top predictions:

        coughing: 98.55%
        sneezing: 1.24%
drinking sipping: 0.15%
       breathing: 0.02%
  brushing teeth: 0.01%
"""
