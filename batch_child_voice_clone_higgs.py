#!/usr/bin/env python3
"""
Batch child voice cloning script for Higgs-Audio
Randomly selects 100 child voice samples and clones them with random English sentences
"""

import os
import sys
import argparse
import random
import torch
import logging
import shutil
from pathlib import Path
from typing import List

import soundfile as sf
import torchaudio
from loguru import logger

# Higgs-Audio imports
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Random English sentence templates for generation
ENGLISH_SENTENCES = [
    "Hello, my name is {name} and I love to play.",
    "The weather is beautiful today, let's go outside.",
    "I really enjoy reading books about adventures.",
    "Can you help me with my homework please?",
    "My favorite color is blue, what's yours?",
    "I want to be a scientist when I grow up.",
    "Let's play hide and seek in the garden.",
    "Do you like ice cream? I love chocolate flavor.",
    "My dog is very friendly and loves to run.",
    "I learned something new at school today.",
    "Can we go to the park this afternoon?",
    "I have a big collection of colorful toys.",
    "My best friend and I like to draw pictures.",
    "The stars are shining brightly in the night sky.",
    "I enjoy listening to music before bedtime.",
    "Let's build a sandcastle at the beach.",
    "I can count to one hundred in English.",
    "My family is planning a trip next summer.",
    "I like to help my mom cook dinner.",
    "The butterfly landed on a beautiful flower.",
    "Can you teach me how to ride a bicycle?",
    "I found a shiny pebble near the river.",
    "My teacher tells us interesting stories every day.",
    "I want to learn how to play the piano.",
    "The rainbow appeared after the rain stopped.",
    "Let's make cookies together this weekend.",
    "I saw a cute rabbit hopping in the field.",
    "My favorite subject at school is art class.",
    "Can we watch a movie tonight after dinner?",
    "I like to water the plants in our garden.",
    "The birds are singing a lovely song outside.",
    "I made a new friend at the playground today.",
    "Let's go on an adventure in the forest.",
    "I can swim very well in the swimming pool.",
    "My grandmother tells the best bedtime stories.",
    "I want to visit the zoo and see elephants.",
    "The moon looks so big and round tonight.",
    "Can you show me how to tie my shoelaces?",
    "I enjoy playing board games with my family.",
    "The wind is blowing gently through the trees.",
    "I like to catch fireflies on summer evenings.",
    "My cat likes to sleep on my warm bed.",
    "Let's go camping and sleep under the stars.",
    "I want to learn more about dinosaurs and fossils.",
    "The ocean waves are crashing on the sandy shore.",
    "I can spell my full name without any mistakes.",
    "My favorite season is spring with all the flowers.",
    "Can we bake a birthday cake for grandpa?",
    "I like to jump in puddles when it rains.",
    "The library has so many wonderful books to read.",
]

EXTENDED_SENTENCES = [
    "I discovered a magical story in the old book.",
    "The kite is flying high up in the clear sky.",
    "I want to become a brave astronaut someday.",
    "My teddy bear is my best companion at night.",
    "Let's explore the mysterious cave behind the hill.",
    "I can hear the gentle sound of rain falling.",
    "The baby bird is learning to fly from the nest.",
    "I enjoy making paper airplanes that fly far.",
    "My bicycle has colorful streamers on the handlebars.",
    "The squirrel is collecting nuts for the winter.",
    "I like to paint pictures of my happy family.",
    "Can you tell me a story about a dragon?",
    "The sunset looks like a beautiful painting tonight.",
    "I want to learn how to speak many languages.",
    "My room is decorated with fun posters and lights.",
    "Let's go ice skating when the pond freezes over.",
    "I can recite my favorite poem from memory.",
    "The garden is full of colorful butterflies today.",
    "I enjoy building tall towers with my blocks.",
    "My hamster runs really fast on the exercise wheel.",
    "The circus came to town with amazing performers.",
    "I like to collect smooth rocks from the beach.",
    "Can we visit the aquarium to see the fish?",
    "The treehouse is my secret hideaway place.",
    "I want to learn magic tricks to amaze everyone.",
    "My snow fort is the biggest one in the neighborhood.",
    "The farmer grows delicious vegetables on the farm.",
    "I enjoy solving puzzles and brain teasers together.",
    "Can you help me find my missing toy car?",
    "The parade has colorful floats and marching bands.",
    "I like to climb trees in the backyard safely.",
    "My backpack is packed for the school field trip.",
    "The chocolate chip cookies smell absolutely delicious.",
    "I want to see the dolphins swimming at the show.",
    "The autumn leaves are turning red and golden.",
    "I can whistle a happy tune when I'm cheerful.",
    "My favorite game is playing tag with friends.",
    "Let's make a snowman with a carrot nose.",
    "The treasure map shows where the gold is buried.",
    "I enjoy riding the carousel at the amusement park.",
    "My sister and I share secrets in our fort.",
    "The lighthouse stands tall by the stormy sea.",
    "I want to learn how to juggle three balls.",
    "Can we go roller skating this Saturday morning?",
    "The baby ducks are following their mother closely.",
    "I like to blow bubbles in the sunny backyard.",
    "My imagination takes me to faraway magical lands.",
    "The thunder sounds loud but I'm not scared.",
    "I enjoy the smell of fresh flowers in spring.",
    "Let's have a picnic by the beautiful lake.",
]

ALL_SENTENCES = ENGLISH_SENTENCES + EXTENDED_SENTENCES


def load_kaldi_files(wav_scp_path, text_tn_path):
    """Load wav.scp and text.tn files"""
    wav_dict = {}
    text_dict = {}

    # Load wav.scp
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    wav_dict[utt_id] = wav_path

    # Load text.tn
    with open(text_tn_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    utt_id, text = parts
                    text_dict[utt_id] = text

    # Find common utterance IDs
    common_ids = set(wav_dict.keys()) & set(text_dict.keys())

    logger.info(f"Loaded {len(wav_dict)} audio files")
    logger.info(f"Loaded {len(text_dict)} text entries")
    logger.info(f"Found {len(common_ids)} matched pairs")

    return wav_dict, text_dict, list(common_ids)


def generate_random_english_sentence():
    """Generate a random English sentence"""
    sentence = random.choice(ALL_SENTENCES)

    # If sentence has placeholder, fill it with a random name
    if "{name}" in sentence:
        names = ["Tom", "Lucy", "Jack", "Emma", "Mike", "Sarah", "David", "Anna"]
        sentence = sentence.format(name=random.choice(names))

    return sentence


def save_sample_directory(
    idx, utt_id, audio_path, original_text, generated_text, output_audio, output_sr, base_output_dir
):
    """
    Save a complete sample in its own subdirectory

    Directory structure:
    sample_0001_<utt_id>/
        ├── prompt_audio.wav          (original audio)
        ├── cloned_audio.wav          (generated audio)
        ├── prompt_text.txt           (original Chinese text)
        └── cloned_text.txt           (generated English text)
    """
    # Create sample subdirectory
    sample_dir_name = f"sample_{idx:04d}_{utt_id}"
    sample_dir = Path(base_output_dir) / sample_dir_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy prompt audio
    prompt_audio_path = sample_dir / "prompt_audio.wav"
    shutil.copy2(audio_path, prompt_audio_path)
    logger.info(f"  → Prompt audio: {prompt_audio_path}")

    # 2. Save cloned audio
    cloned_audio_path = sample_dir / "cloned_audio.wav"
    sf.write(str(cloned_audio_path), output_audio, output_sr)
    logger.info(f"  → Cloned audio: {cloned_audio_path}")

    # 3. Save prompt text
    prompt_text_path = sample_dir / "prompt_text.txt"
    with open(prompt_text_path, "w", encoding="utf-8") as f:
        f.write(original_text)
    logger.info(f"  → Prompt text: {prompt_text_path}")

    # 4. Save cloned text
    cloned_text_path = sample_dir / "cloned_text.txt"
    with open(cloned_text_path, "w", encoding="utf-8") as f:
        f.write(generated_text)
    logger.info(f"  → Cloned text: {cloned_text_path}")

    return sample_dir


def batch_clone(args, serve_engine, wav_dict, text_dict, selected_ids):
    """Batch process voice cloning using Higgs-Audio"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a CSV file to record results
    results_csv = output_dir / "clone_results.csv"
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write("ID,Original_Audio,Original_Text,Generated_Text,Output_Directory,Status\n")

    success_count = 0
    failed_count = 0

    for idx, utt_id in enumerate(selected_ids, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {idx}/{len(selected_ids)}: {utt_id}")
        logger.info(f"{'=' * 80}")

        try:
            # Get audio path and original text
            audio_path = wav_dict[utt_id]
            original_text = text_dict[utt_id]

            # Check if audio file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                failed_count += 1
                continue

            # Generate random English sentence
            generated_text = generate_random_english_sentence()

            logger.info(f"Original text: {original_text}")
            logger.info(f"Generated text: {generated_text}")
            logger.info(f"Audio path: {audio_path}")

            # Prepare messages for voice cloning (similar to examples/generation.py)
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                "<|scene_desc_end|>"
            )

            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                # First, provide the voice prompt (original audio + text)
                Message(
                    role="user",
                    content=original_text,
                ),
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=audio_path),
                ),
                # Then, ask to clone with new text
                Message(
                    role="user",
                    content=generated_text,
                ),
            ]

            # Perform voice cloning
            logger.info("[INFO] Starting inference...")
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                seed=args.seed,
            )

            if output.audio is not None and len(output.audio) > 0:
                # Save complete sample in subdirectory
                sample_dir = save_sample_directory(
                    idx,
                    utt_id,
                    audio_path,
                    original_text,
                    generated_text,
                    output.audio,
                    output.sampling_rate,
                    args.output_dir,
                )

                # Record success
                with open(results_csv, "a", encoding="utf-8") as f:
                    f.write(f"{utt_id},{audio_path},{original_text},{generated_text},{sample_dir},SUCCESS\n")

                success_count += 1
                logger.info(f"✓ Successfully cloned and saved to: {sample_dir.name}")
            else:
                logger.error(f"✗ Clone failed for {utt_id}: Empty audio output")
                failed_count += 1
                with open(results_csv, "a", encoding="utf-8") as f:
                    f.write(f"{utt_id},{audio_path},{original_text},{generated_text},,FAILED\n")

        except Exception as e:
            logger.error(f"✗ Error processing {utt_id}: {str(e)}")
            failed_count += 1
            with open(results_csv, "a", encoding="utf-8") as f:
                f.write(
                    f"{utt_id},{audio_path if 'audio_path' in locals() else 'N/A'},{original_text if 'original_text' in locals() else 'N/A'},,,ERROR: {str(e)}\n"
                )

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Batch processing completed!")
    logger.info(f"Total: {len(selected_ids)} | Success: {success_count} | Failed: {failed_count}")
    logger.info(f"Results saved to: {results_csv}")
    logger.info(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Batch child voice cloning for Higgs-Audio")

    # Data paths
    parser.add_argument(
        "--wav-scp",
        type=str,
        default="/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/wav.scp",
        help="Path to wav.scp file",
    )
    parser.add_argument(
        "--text-tn",
        type=str,
        default="/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/text.tn",
        help="Path to text.tn file",
    )

    # Model paths
    parser.add_argument(
        "--model-path", type=str, default="bosonai/higgs-audio-v2-generation-3B-base", help="Higgs-Audio model path"
    )
    parser.add_argument(
        "--audio-tokenizer-path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Higgs-Audio tokenizer path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./child_voice_clone_output_higgs", help="Output directory for cloned audio"
    )

    # Sampling parameters
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to randomly select (default: 100)"
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--seed", type=int, default=1988, help="Model random seed")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.seed)

    # Load kaldi files
    logger.info("Loading kaldi files...")
    wav_dict, text_dict, common_ids = load_kaldi_files(args.wav_scp, args.text_tn)

    # Randomly select samples
    if len(common_ids) < args.num_samples:
        logger.warning(f"Requested {args.num_samples} samples but only {len(common_ids)} available")
        args.num_samples = len(common_ids)

    selected_ids = random.sample(common_ids, args.num_samples)
    logger.info(f"Randomly selected {len(selected_ids)} samples")

    # Load Higgs-Audio model
    logger.info(f"Loading Higgs-Audio model from: {args.model_path}")
    try:
        serve_engine = HiggsAudioServeEngine(args.model_path, args.audio_tokenizer_path, device=args.device)
        logger.info("✓ Higgs-Audio model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        exit(1)

    # Batch process
    logger.info("\nStarting batch voice cloning...")
    batch_clone(args, serve_engine, wav_dict, text_dict, selected_ids)


if __name__ == "__main__":
    main()
