from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
import torch
import scipy

def generate_audio():
    # Constants
    NUM_GENERATIONS = 3
    SAMPLE_RATE = 16000

    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    negative_prompt = "Low quality, average quality."
    prompt = "The sound of Brazilian samba drums with waves gently crashing in the background"

    generator = torch.Generator("cuda").manual_seed(0)

    for i in range(NUM_GENERATIONS):
        audio = pipe(
            prompt,
            num_inference_steps=20,
            negative_prompt=negative_prompt,
            audio_length_in_s=10.24,
            generator=generator,
        ).audios[0]

        # Use f-strings for string formatting
        file_name = f"generated/{i}.mp3"
        scipy.io.wavfile.write(file_name, rate=SAMPLE_RATE, data=audio)

def main():
    generate_audio()

if __name__ == "__main__":
    main()
