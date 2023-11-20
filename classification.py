import torch.nn.functional as F
import json
from msclap import CLAP
from collections import defaultdict 

feature_classes = [['fast rhythm','slow rhythm'],['high pitch','low pitch'],['loud intensity','soft intensity'],['bright timbre','mellow timbre']]
eval_classes = [['high arousal','low arousal'],['positive valence','negative valence']]

result_dict = {}
prompt = 'music with '

n = 251

for classes in feature_classes:
    
    class_prompts = []
    
    for desc in classes:
        class_prompts.append(prompt + desc)
    
    print(class_prompts)
    
    result = defaultdict(float)
    
    for i in range(n):

        audio_files = []
        music = 'music/music (' + str(i+1) + ').mp3'
        audio_files.append(music)
        # Load and initialize CLAP
        # Setting use_cuda = True will load the model on a GPU using CUDA
        clap_model = CLAP(version = '2023', use_cuda=True)

        # compute text embeddings from natural text
        text_embeddings = clap_model.get_text_embeddings(class_prompts)

        # compute the audio embeddings from an audio file
        audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)

        # compute the similarity between audio_embeddings and text_embeddings
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

        similarity = F.softmax(similarity, dim=1)
        values, indices = similarity[0].topk(2)

        for value, index in zip(values, indices):
             #print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
            result[classes[index]] += 100 * value.item()
        print("{}th music finish".format(i+1))
    
    for k, v in result.items():
        result_dict[k] = round(v/n, 2)
    
with open("result.json", 'w') as f:
  json.dump(result_dict, f)
