# NUMonomer

<p align="center">
  <img src="utils/NUMonomer_overview.jpg" alt="Overview of the NUMonomer framework" width="900">
</p>

<p align="center">
  <em>Overview of the NUMonomer framework.</em>
</p>

NUMonomer is an accurate, efficient, and scalable end-to-end deep-learning framework for predicting the three-dimensional structures of **RNA** and **single-stranded DNA (ssDNA)** directly from their primary sequences. By eliminating the need for auxiliary inputs and employing a highly scalable architecture, NUMonomer can predict the structure of a 5,000-nucleotide nucleic acid sequence in approximately 20 seconds on a single NVIDIA H100 PCIe GPU. This efficiency makes it well suited for both long-sequence modeling and large-scale structure prediction. This repository provides the official implementation of the NUMonomer inference pipeline.

## Contents

- [Installation](#installation)
- [Optional acceleration](#optional-acceleration)
- [Repository structure](#repository-structure)
- [Input format](#input-format)
- [Quick start](#quick-start)
- [Command-line arguments](#command-line-arguments)
- [Output files](#output-files)
- [Prediction selection and confidence](#prediction-selection-and-confidence)
- [Long-sequence inference](#long-sequence-inference)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Installation

### 1. Create a Conda environment

```bash
conda create -n numonomer python=3.12 -y
conda activate numonomer
```

### 2. Install dependencies

Install a PyTorch build compatible with your CUDA driver, and then install the remaining dependencies:

```bash
pip install torch
pip install biopython numpy ml-collections
```

An example tested software environment is:

- Python 3.12
- PyTorch 2.11
- Biopython 1.87
- NumPy 2.4.3
- ml-collections 1.1.0

> GPU inference is strongly recommended. The inference script defaults to `cuda:0` and uses `bfloat16` mixed precision. CPU inference may be substantially slower and may require code changes depending on the supported PyTorch operations.

### 3. Clone the repository

```bash
git clone https://github.com/yunda-si/NUMonomer.git
cd NUMonomer
```

### 4. Download model weights

Download the pretrained NUMonomer model weights from Google Drive: [Download NUMonomer weights](https://drive.google.com/drive/folders/1K9fG3ndV2UH3atwyrxHDqPN-oHyvhRwG?usp=sharing)

Place the downloaded checkpoint in the `weights/` directory, for example:

```text
weights/NUMonomer.pt
```

## Optional acceleration

The confidence module uses native attention computation by default. When supported by the local CUDA and PyTorch environment, optional attention kernels can reduce GPU memory usage and improve inference speed.

Possible acceleration options include:

- [FlashAttention](https://github.com/dao-ailab/flash-attention);
- [DS4Sci EvoformerAttention from DeepSpeed](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/);
- PyTorch model compilation.

After installing the required optional dependencies, the corresponding settings can be enabled in the config.py, for example:

```python
config.inference.use_dsattn = True
config.inference.disable_compile = False
```
## Repository structure

```text
NUMonomer/
├── config.py
├── dataset.py
├── LICENSE
├── model.py
├── modules.py
├── prediction.py
├── README.md
├── structure_module.py
├── __init__.py
│
├── example/
│   ├── test.fasta
│   └── test.pdb
│
├── np/
│   ├── residue_constants.py
│   └── __init__.py
│
├── utils/
│   ├── check_input.py
│   ├── rigid_utils.py
│   ├── save_struc.py
│   ├── seq_utils.py
│   ├── NUMonomer_overview.png
│   └── __init__.py
│
└── weights/
    └── NUMonomer.pt
```

## Input format

NUMonomer accepts either:

1. a single sequence file through `--seq_file`; or
2. a directory containing sequence files through `--seq_path`.

The two options are mutually exclusive. Exactly one must be provided.

Each FASTA header must use the following format:

```text
>CHAIN_ID|MOLECULE_TYPE
```

where:

- `CHAIN_ID` is the chain identifier used in the output structure;
- `MOLECULE_TYPE` must be either `rna` or `dna`.

### RNA example

```fasta
>A|rna
GGGAGACCGGAAUUCUGGUCCGAGUAGAGUGUGAGCUCCGUAACUAGUCGCGU
```

### ssDNA example

```fasta
>A|dna
GGGAGACCGGAATTCTGGTCCGAGTAGAGTGTGAGCTCCGTAACTAGTCGCGT
```


## Quick start

### Predict one target

```bash
python -u prediction.py \
  --seq_file ./example/test.fasta \
  --save_path ./results \
  --weight ./weights/NUMonomer.pt \
```

### Predict all targets in a directory

```bash
python -u prediction.py \
  --seq_path ./example \
  --save_path ./results \
  --weight ./weights/NUMonomer.pt
```

The script validates the supplied input files before constructing the inference dataset.

## Command-line arguments

| Argument | Default | Required | Description |
|---|---:|:---:|---|
| `-seq_file` | `None` | No* | Path to one input sequence file. Mutually exclusive with `--seq_path`. |
| `-seq_path` | `None` | No* | Directory containing input sequence files. Mutually exclusive with `--seq_file`. |
| `-save_path` | `None` | Yes | Directory in which target-specific output folders are created. |
| `-weight` | `None` | Yes | Path to the pretrained model checkpoint. |
| `-ftype` | `cif` | No | Requested structure serialization format: `cif` or `pdb`. |
| `-device` | `cuda:0` | No | PyTorch device used for inference. |
| `-seed` | `42` | No | Random seed for Python, NumPy, and PyTorch. |
| `-last` | disabled | No | Export the final recycling iteration instead of the confidence-selected iteration. |
| `-ncpu` | `8` | No | Number of CPU threads and post-processing workers. |
| `-split_seq` | `0` | No | Enable sequence-dimension chunking to reduce peak memory usage. `0` disables chunking. |
| `-split_atom` | `0` | No | Enable atom-dimension chunking to reduce peak memory usage. `0` disables chunking. |
| `-num_iter` | `8` | No | Number of structure-recycling iterations. |
| `-clamp_plddt` | `512` | No | Number of leading residues used for the confidence calculation. |

\* Supply either `--seq_file` or `--seq_path`, but not both.

## Output files

For each target, NUMonomer creates a target-specific directory under `--save_path`.

A typical output layout is:

```text
results/
└── <target_name>/
    ├── pred_NUMonomer_42.cif
    └── log_NUMonomer_42.json
```

The exact filename contains:

- the checkpoint filename stem;
- the random seed;
- the selected structure format.

The structure file contains the predicted atomic coordinates. Per-residue confidence values are written to the B-factor field by the structure writer.

The JSON log contains fields similar to:

```json
{
  "timing": " 12.34",
  "plddt": [" 72.1", " 74.8", " 77.3", " 78.0"],
  "len_seq": "   256",
  "target": "example_target",
  "idx_coords": [3]
}
```

## Prediction selection and confidence

By default, the script evaluates the mean predicted pLDDT score across recycling iterations and exports a confidence-selected structure.

To export the final recycling iteration instead, add `--last`:

```bash
python -u prediction.py \
  --seq_file ./example/test.fasta \
  --save_path ./results \
  --weight ./weights/NUMonomer.pt \
  --last
```

Using `--last` can reduce confidence-selection overhead, but the final iteration is not necessarily the iteration with the highest predicted confidence.

> **Implementation note:** the current confidence-selection code assumes that at least four recycling outputs are available. Keep `--num_iter` at `4` or higher when using the default confidence-based selection. When using fewer iterations, enable `--last` or update the selection logic accordingly.

## Long-sequence inference

NUMonomer is designed to process sequences spanning thousands of nucleotides. Peak memory usage nevertheless depends on sequence length, model configuration, recycling iterations, attention implementation, and hardware.

Chunked inference can be enabled with:

```bash
--split_seq <positive_integer>
--split_atom <positive_integer>
```

Both options default to `0`, which disables chunking and generally provides the highest throughput at the cost of greater peak memory usage. Positive values enable chunked computation and can reduce GPU memory consumption, although inference may become slower.

Example for a long target:

```bash
python -u prediction.py \
  --seq_file ./example/long_target.fasta \
  --save_path ./results_long \
  --weight ./weights/NUMonomer.pt \
  --split_seq 128 \
  --split_atom 2 \
  --num_iter 8 \
  --clamp_plddt 512
```

Additional memory-saving strategies include:

- enabling `--last`;
- reducing `--num_iter`, while keeping the implementation note above in mind;
- adjusting `--clamp_plddt`;
- enabling a supported memory-efficient attention backend.

These changes can affect speed, confidence estimation, or prediction quality and should be validated for the intended application.

## Reproducibility

The `--seed` option initializes random-number generators for:

- Python's `random` module;
- NumPy;
- PyTorch CPU operations; and
- PyTorch CUDA operations.


Exact reproducibility can still depend on the PyTorch and CUDA versions, GPU hardware, compiler settings, and nondeterministic kernels.

## Citation

The complete citation and BibTeX entry will be added after the manuscript becomes publicly available.

```bibtex
@article{NUMonomer2026,
  title   = {NUMonomer enables accurate and efficient nucleic acid structure prediction from primary sequence alone},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026},
  doi     = {To be updated}
}
```

If you have any questions or feedback, please open a GitHub issue or feel free to contact us at [yunda_si@ucas.edu.cn](mailto:yunda_si@ucas.edu.cn) or [lnchen@sjtu.edu.cn](mailto:lnchen@sjtu.edu.cn).
