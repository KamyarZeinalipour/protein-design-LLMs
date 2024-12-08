# Protein Sequence Generation using Large Language Models

This repository contains a Python script designed to generate protein sequences using fine-tuned large language models (LLMs). The script leverages models such as Mistral-7B, Llama-2-7B, Llama-3-8B, and Gemma-7B, which have been adapted for protein design tasks. This code is part of the research presented in the paper:

**Design Proteins Using Large Language Models: Enhancements and Comparative Analyses**

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Models and Tokenizers](#models-and-tokenizers)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Results](#results)
- [Reference](#reference)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Recent advancements in natural language processing (NLP) have shown that large language models (LLMs) can be adapted for structured biological data such as protein sequences. Proteins, composed of sequences of amino acids, have properties analogous to languages, with "grammar" defined by biochemical interactions.

This repository provides a script that demonstrates how LLMs can generate biologically plausible protein sequences even with limited training data (~42,000 human protein sequences). By fine-tuning pre-trained LLMs and adapting tokenizers to protein data, we achieve efficient protein design comparable to specialized models trained on larger datasets.

---

## Features

- **Multiple Model Support**: Generate protein sequences using different fine-tuned LLMs.
- **Custom Tokenizers**: Utilizes tokenizers retrained specifically for protein sequences.
- **Dynamic BOS and EOS Tokens**: Automatically handles model-specific Beginning-of-Sequence (BOS) and End-of-Sequence (EOS) tokens.
- **Configurable Generation Parameters**: Adjust temperature, sequence length, and number of generations.
- **Result Output**: Saves accepted protein sequences to a CSV file for further analysis.

---

## Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended for performance)
- Python Libraries:
  - torch
  - transformers
  - pandas
  - argparse

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

   Install required packages:

   ```bash
   pip install torch transformers pandas argparse
   ```

3. **Verify CUDA Installation (Optional)**

   If you have a CUDA-compatible GPU, ensure that PyTorch recognizes it:

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

   This should output `True` if CUDA is available.

---

## Models and Tokenizers

The script supports the following models, each with its own tokenizer and special tokens:

1. **P-gemma-7B**
   - **Model Repository**: `Kamyar-zeinalipour/P-gemma-7B`
   - **Tokenizer Repository**: `Kamyar-zeinalipour/protein-tokenizer-gemma`
   - **BOS Token**: `<bos>`
   - **EOS Token**: `<eos>`

2. **P-Mistral**
   - **Model Repository**: `Kamyar-zeinalipour/P-Mistral-7B`
   - **Tokenizer Repository**: `Kamyar-zeinalipour/protein-tokenizer-mistral`
   - **BOS Token**: `<s>`
   - **EOS Token**: `</s>`

3. **P-Llama2-7B**
   - **Model Repository**: `Kamyar-zeinalipour/P-Llama2-7B`
   - **Tokenizer Repository**: `Kamyar-zeinalipour/protein-tokenizer-llama2`
   - **BOS Token**: `<s>`
   - **EOS Token**: `</s>`

4. **P-Llama3-8B**
   - **Model Repository**: `Kamyar-zeinalipour/P-Llama3-8B`
   - **Tokenizer Repository**: `Kamyar-zeinalipour/protein-tokenizer-llama3`
   - **BOS Token**: `<|begin_of_text|>`
   - **EOS Token**: `<|end_of_text|>`

---

## Usage

The script `generate_proteins.py` generates protein sequences using specified models and parameters. It can be run from the command line with customizable arguments.

### Command-Line Arguments

- `--model_name`: **(Required)** Name of the model to use.
  - Options: `P-gemma-7B`, `P-Mistral`, `P-Llama2-7B`, `P-Llama3-8B`.
- `--num_generations`: **(Required)** Number of protein sequences to generate.
- `--temperature`: Sampling temperature for text generation. Default is `0.8`.
- `--min_length`: Minimum length of valid protein sequences. Default is `25`.
- `--max_length`: Maximum length of valid protein sequences. Default is `150`.
- `--output_file`: Name of the CSV file to save accepted sequences. Default is `Accepted_Texts.csv`.

### Examples

1. **Generate 10 Sequences with P-Mistral**

   ```bash
   python generate_proteins.py --model_name P-Mistral --num_generations 10
   ```

2. **Generate Sequences with Custom Length and Temperature**

   ```bash
   python generate_proteins.py --model_name P-Llama2-7B --num_generations 20 --temperature 0.7 --min_length 50 --max_length 200 --output_file llama2_proteins.csv
   ```

3. **Using P-Llama3-8B Model**

   ```bash
   python generate_proteins.py --model_name P-Llama3-8B --num_generations 15 --output_file llama3_proteins.csv
   ```

---

## Results

- The script generates the specified number of protein sequences.
- Each sequence is cleaned by removing the model-specific BOS and EOS tokens.
- Sequences not meeting the length criteria are discarded.
- Accepted sequences are printed to the console and saved in the specified CSV file.

Example output in the console:

```
Accepted: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF...
Accepted: GQTFYVDGAQLFAVRMKGIPKLVQPQAKEMGLMR...
...
Accepted text saved to Accepted_Texts.csv
```

---

## Reference

### Paper: Design Proteins Using Large Language Models: Enhancements and Comparative Analyses

The code in this repository is part of the research conducted in the aforementioned paper. The study explores the adaptation of large language models for protein sequence generation, aiming to bridge computational biology and NLP.

#### Abstract

*The paper investigates the use of LLMs for protein sequence generation, utilizing models like Mistral-7B, Llama-2-7B, Llama-3-8B, and Gemma-7B. It demonstrates that LLMs can generate biologically plausible protein sequences even when trained on smaller datasets (~42,000 human protein sequences). Key findings include efficient performance with limited data, structural accuracy of generated proteins, and the impact of model architecture on biological tasks.*

#### Key Findings

- **Efficient Performance with Limited Data**: LLMs performed comparably to models trained on millions of sequences.
- **Structural Accuracy**: High-confidence protein structures were generated and validated using tools like AlphaFold 2.
- **Model Diversity Matters**: Different architectures and fine-tuning strategies significantly influenced performance.
- **Open-Source Contribution**: Trained models and datasets have been made publicly available.

#### Implications

- **Drug Discovery**: Potential for novel protein design in pharmaceutical research.
- **Understanding Protein Structure**: Advances knowledge in protein structure-function relationships.
- **Accessibility**: Provides powerful tools for researchers with limited computational resources.

#### Access

The paper is available on [arXiv](https://arxiv.org/abs/2408.06396) and [aclanthology](https://aclanthology.org/2024.langmol-1.5/). The models and datasets are hosted on Hugging Face and GitHub.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.txt) file for details.

---

## Acknowledgements

- **Authors**: Kamyar Zeinalipour, Neda Jamshidi, Monica Bianchini, Marco Maggini and Marco Gori
- **Affiliation**: University of Siena
- **Contact**: kamyar.zeinalipour2@unisi.it

---

**Note**: Ensure you have the necessary permissions and access rights to download and use the models from Hugging Face. Some models may require authentication or acceptance of specific terms and conditions.

---

## Troubleshooting

- **CUDA Errors**: If you encounter issues related to CUDA, ensure your GPU drivers are up to date and that PyTorch is correctly installed with CUDA support.
- **Memory Issues**: Large models may require significant GPU memory. If you run into memory errors, consider reducing the batch size or using a smaller model.
- **Model Not Found**: Verify that the model and tokenizer repositories exist and are correctly spelled in the `MODEL_INFO` dictionary within the script.

---

## Further Reading

- **AlphaFold 2**: For protein structure prediction and validation.
- **Rosetta Relax**: For analyzing energy profiles of protein structures.
- **Hugging Face Transformers**: Documentation on model and tokenizer usage.

---

## Contributions

Contributions are welcome! If you'd like to contribute to this project, please open an issue or submit a pull request.

---

## Citation

If you use this code or the models in your research, please cite the paper:

```
@inproceedings{zeinalipour-etal-2024-design,
    title = "Design Proteins Using Large Language Models: Enhancements and Comparative Analyses",
    author = "Zeinalipour, Kamyar  and
      Jamshidi, Neda  and
      Bianchini, Monica  and
      Maggini, Marco  and
      Gori, Marco",
    editor = "Edwards, Carl  and
      Wang, Qingyun  and
      Li, Manling  and
      Zhao, Lawrence  and
      Hope, Tom  and
      Ji, Heng",
    booktitle = "Proceedings of the 1st Workshop on Language + Molecules (L+M 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.langmol-1.5",
    doi = "10.18653/v1/2024.langmol-1.5",
    pages = "34--47",
    abstract = "Pre-trained LLMs have demonstrated substantial capabilities across a range of conventional natural language processing (NLP) tasks, such as summarization and entity recognition. In this paper, we explore the application of LLMs in the generation of high-quality protein sequences. Specifically, we adopt a suite of pre-trained LLMs, including Mistral-7B, Llama-2-7B, Llama-3-8B, and gemma-7B, to produce valid protein sequences. All of these models are publicly available (https://github.com/KamyarZeinalipour/protein-design-LLMs).Unlike previous work in this field, our approach utilizes a relatively small dataset comprising 42,000 distinct human protein sequences. We retrain these models to process protein-related data, ensuring the generation of biologically feasible protein structures. Our findings demonstrate that even with limited data, the adapted models exhibit efficiency comparable to established protein-focused models such as ProGen varieties, ProtGPT2, and ProLLaMA, which were trained on millions of protein sequences. To validate and quantify the performance of our models, we conduct comparative analyses employing standard metrics such as pLDDT, RMSD, TM-score, and REU. Furthermore, we commit to making the trained versions of all four models publicly available, fostering greater transparency and collaboration in the field of computational biology.",
}
```

---

**Disclaimer**: This code is for research purposes. Generated protein sequences should be validated experimentally before any practical application.

---

**Thank you for using our protein generation script! If you have any questions or feedback, please reach out.**
