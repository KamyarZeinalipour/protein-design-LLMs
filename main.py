import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_code_completion(prompt, temperature, model, tokenizer):
    """
    Generates text based on the input prompt using the specified temperature.
    """
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=500,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def generate_and_save_text(num_accepted_texts, temperature, min_length, max_length, output_file, model, tokenizer, bos_token, eos_token):
    """
    Generates and saves text to a CSV file based on the given parameters.
    """
    accepted_texts = []
    total_generations = 0

    while len(accepted_texts) < num_accepted_texts:
        # Use the model's BOS token as the prompt
        generated_text = get_code_completion('', temperature, model, tokenizer)

        # Clean up the generated text by removing BOS and EOS tokens
        generated_text = generated_text.replace(bos_token, '', 1).replace(eos_token, '', 1).strip()

        # Filter by character length
        if min_length <= len(generated_text) <= max_length:
            accepted_texts.append(generated_text)
            print(f"Generated Protein: {generated_text}")

        total_generations += 1

    print(f"Total text generations attempted: {total_generations}")

    # Save to a CSV file
    df = pd.DataFrame({'Generated Protein': accepted_texts})
    df.to_csv(output_file, index=False)
    print(f"Accepted text saved to {output_file}")

if __name__ == "__main__":
    # Map of model names to their repositories and tokens
    MODEL_INFO = {
        'P-gemma-7B': {
            'model_repo': 'Kamyar-zeinalipour/P-gemma-7B',
            'tokenizer_repo': 'Kamyar-zeinalipour/protein-tokenizer-gemma',
            'bos_token': '<bos>',
            'eos_token': '<eos>'
        },
        'P-Mistral-7B': {
            'model_repo': 'Kamyar-zeinalipour/P-Mistral-7B',
            'tokenizer_repo': 'Kamyar-zeinalipour/Mistral-tokenizer-prot',
            'bos_token': '<s>',
            'eos_token': '</s>'
        },
        'P-Llama2-7B': {
            'model_repo': 'Kamyar-zeinalipour/P-Llama2-7B',
            'tokenizer_repo': 'Kamyar-zeinalipour/protein-tokenizer-llama2',
            'bos_token': '<s>',
            'eos_token': '</s>'
        },
        'P-Llama3-8B': {
            'model_repo': 'Kamyar-zeinalipour/P-Llama3-8B',
            'tokenizer_repo': 'Kamyar-zeinalipour/protein-tokenizer-llama3',
            'bos_token': '<|begin_of_text|>',
            'eos_token': '<|end_of_text|>'
        }
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate text samples using a language model.")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Name of the model to use (e.g., P-gemma-7B).")
    parser.add_argument('--num_generations', type=int, required=True,
                        help="Number of acceptable text samples to generate.")
    parser.add_argument('--temperature', type=float, default=0.8,
                        help="Sampling temperature (default: 0.8).")
    parser.add_argument('--min_length', type=int, default=25,
                        help="Minimum length of valid text samples (default: 25).")
    parser.add_argument('--max_length', type=int, default=150,
                        help="Maximum length of valid text samples (default: 150).")
    parser.add_argument('--output_dir', type=str, default=".",
                        help="Output directory (default: current directory).")

    args = parser.parse_args()

    # Retrieve model information based on provided model name
    if args.model_name not in MODEL_INFO:
        raise ValueError(f"Model name '{args.model_name}' not found in model information.")

    model_info = MODEL_INFO[args.model_name]

    model_repo = model_info['model_repo']
    tokenizer_repo = model_info['tokenizer_repo']
    bos_token = model_info['bos_token']
    eos_token = model_info['eos_token']

    # Load tokenizer and set BOS and EOS tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        quantization_config=None,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if not hasattr(model, "hf_device_map"):
        model.cuda()

    # Create output file path with model name
    output_file = os.path.join(args.output_dir, f"Generated_Proteins_{args.model_name}.csv")

    # Generate and save text
    generate_and_save_text(
        num_accepted_texts=args.num_generations,
        temperature=args.temperature,
        min_length=args.min_length,
        max_length=args.max_length,
        output_file=output_file,
        model=model,
        tokenizer=tokenizer,
        bos_token=bos_token,
        eos_token=eos_token,
    )
