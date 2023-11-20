import torch.nn.functional as F
import json
from msclap import CLAP
from collections import defaultdict

def get_audio_files_gt(n):
    audio_files = []
    if n > 300:
        for i in range(0, n, 3):
            music = f'music/music ({i+1}).mp3'
            audio_files.append(music)
    else:
        for i in range(n):
            music = f'music/music ({i+1}).mp3'
            audio_files.append(music)
    return audio_files

def get_audio_files_generated(n):
    audio_files = []
    for i in range(n):
        music = f'generated/{i}.mp3'
        audio_files.append(music)
    return audio_files

def process_audio_files(audio_files, class_prompts):
    clap_model = CLAP(version='2023', use_cuda=True)
    text_embeddings = clap_model.get_text_embeddings(class_prompts)
    audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)
    return clap_model.compute_similarity(audio_embeddings, text_embeddings)

def feature():
    feature_classes = [['fast rhythm', 'slow rhythm'], ['high pitch', 'low pitch'], ['loud intensity', 'soft intensity'], ['bright timbre', 'mellow timbre']]
    eval_classes = [['high arousal', 'low arousal'], ['positive valence', 'negative valence']]
    result_dict = defaultdict(float)
    prompt = 'music with '

    num_music = 225
    audio_files = get_audio_files_gt(num_music)

    for classes in feature_classes:
        class_prompts = [prompt + desc for desc in classes]
        print(class_prompts)
        for i in range(len(audio_files)):
            audio_file = []
            audio_file.append(audio_files[i])
            similarity = process_audio_files(audio_file, class_prompts)
            similarity = F.softmax(similarity, dim=1)
            values, indices = similarity[0].topk(2)
            for value, index in zip(values, indices):
                result_dict[classes[index]] += 100 * value.item()
            print(f"{i+1}th music finish")

    for k, v in result_dict.items():
        result_dict[k] = round(v / len(audio_files), 2)

    with open("result.json", 'w') as f:
        json.dump(result_dict, f)

def eval():
    eval_classes = [['high arousal', 'low arousal'], ['positive valence', 'negative valence']]
    prompt = 'music with '
    result_list = []
    num_music = 3
    audio_files = get_audio_files_generated(num_music)
    for i in range(len(audio_files)):
            result_dict = defaultdict(float)
            audio_file = []
            audio_file.append(audio_files[i])
            for classes in eval_classes:
                class_prompts = [prompt + desc for desc in classes]
                print(class_prompts)
                similarity = process_audio_files(audio_file, class_prompts)
                similarity = F.softmax(similarity, dim=1)
                values, indices = similarity[0].topk(2)
                for value, index in zip(values, indices):
                    result_dict[classes[index]] = 100 * value.item()
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
    #feature()
    eval()
