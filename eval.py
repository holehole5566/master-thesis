import json
from collections import defaultdict
import torch.nn.functional as F
from feature import process_audio_files

def get_audio_files_generated(n):
    audio_files = []
    for i in range(n):
        music = f'generated/{i}.mp3'
        audio_files.append(music)
    return audio_files

def eval():
    classes = ["happy song", "sad song", "angry song", "relaxed song", "excited song", "pleased song", "annoying song", "nervous song", "bored song", "sleepy song", "calm song", "peaceful song"]
    prompt = 'This is a '
    result_list = []
    num_music = 225
    audio_files = get_audio_files_generated(num_music)
    for i in range(len(audio_files)):
        result_dict = defaultdict(float)
        audio_file = []
        audio_file.append(audio_files[i])
        class_prompts = [prompt + desc for desc in classes]
        similarity = process_audio_files(audio_file, class_prompts)
        similarity = F.softmax(similarity, dim=1)
        values, indices = similarity[0].topk(len(classes))
        for value, index in zip(values, indices):
            result_dict[classes[index]] = value.item()
        for k, v in result_dict.items():
            result_dict[k] = round(v , 2)
        result_list.append(result_dict)
        print(f"{i+1}th music finish")
    
    results = {
        "results": result_list
    }
    
    with open("result_generated.json", 'w') as f:

        json.dump(results, f)

if __name__ == "__main__":
    eval()