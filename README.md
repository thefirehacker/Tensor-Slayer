# Tensor Slayer

## Reverse engineering LLMs using LLMs 

Today, the AI frontier labs have a strong compute hegemony. Consumer-grade hardware is often left ignored, specifically when it comes to model interpretability and improving models for specific tasks. Techniques such as fine-tuning are still compute-hungry, non-cost effective, and most importantly, require curated datasets for achieving significant improvements over the base model. And, last but not the least, fine-tuning requires hours if not days - in order to get it right. You are typically required to create a standard dataset, curate it, and THEN train your models. 

## What 

This framework introduces faster way of making models better, *in matter of minutes**, compared to the base model. 

## How

- Essentially using powerful LLM to reverse engineer the target LLM.


Taking a novel approach of editing binary files directly by leveraging an agentic framework - to improve relative model performance by up to 30% from the base model:

- No fine-tuning
- No inference
- No compute cost
- No dataset
- No wasting of time. Improving model takes 2-5 minutes. 
- Currently supports improvments in Math, Reasoning, and STEM.

Further, adding a novel approach to perform static analysis on binary files of the models for better interpretability - this kind of interpretability is not meant to be a replacement of inference-based interpretability but to rather act as complementary.

## Benchmark performance test on Qwen3_0.6B model 

- The script has been extensively used to improve performance of a Qwen3_0.6B, a tiny reasoning model by the behemoth Alibaba cloud.
- In the MMLU benchmark tests below, Qwen_0.6B is the base model and subsequent Qwen_T* models are iterative upgrades. Each upgrade has led to incremental improvement in the model performance and ultimate leading to 25% relative performance      improvement compared to the base model - without fine tuning. 

**STEM**
![category_comparison](https://github.com/user-attachments/assets/c6b557eb-ce8e-4ed9-ad3c-4423e893baeb)

**HEATMAP** 
![category_heatmap (Copy 4)](https://github.com/user-attachments/assets/16039a02-02fa-436a-a994-6bb650be861c)



## Pain Points 

### Democratizing Model Enhancement
- **High Compute Barriers**: Most model improvement techniques are inaccessible to individuals or organizations without massive compute resources
- **Fine-tuning Inefficiency**: Traditional fine-tuning requires extensive datasets, expertise, and time
- **Performance Plateaus**: Getting the last 10-30% of task-specific performance often costs exponentially more

### Advancing Model Interpretability
- **Black Box Models**: Modern AI models operate as inscrutable black boxes
- **Inference-Dependent Analysis**: Most interpretability tools require running inference, which is compute-intensive
- **Limited Edit Capabilities**: Few tools allow for targeted modifications to address discovered issues

## Solutions

### Direct Tensor Patching
By focusing on the precise tensors that influence specific behaviors, we enable targeted modifications that can yield significant performance improvements without the overhead of fine-tuning.

### Static Binary Analysis
Our innovative approach allows you to understand model structure and behavior patterns by analyzing the binary weights directly, providing insights normally only available through extensive inference testing.

## Overview

Tensor Slayer is an advanced interactive tool designed for investigating, analyzing, and modifying machine learning models at the tensor level. This tool enables AI researchers and engineers to explore model internals, diagnose issues, and apply targeted modifications to improve model performance without requiring full retraining.

## Features

- **Interactive Model Exploration**: Browse and visualize the structure of machine learning models, with special focus on different tensor types (attention mechanisms, MLP layers, embeddings).

- **Advanced Tensor Analysis**: Perform comprehensive analysis of individual tensors including:
  - Detailed statistics (min, max, mean, standard deviation, sparsity)
  - Percentile distributions
  - Pattern detection
  - Value sampling
  - Operation simulation

- **Guided Patching**: Apply modifications to model tensors through an interactive session that includes:
  - Operation selection (scale, add, normalize)
  - Value and target specification
  - Change preview with before/after statistics
  - Safety checks
  - Automatic model backup

- **Tensor Investigation**: Query-based exploration of model components with targeted analysis for:
  - Specific tensor types
  - Token embeddings
  - Behavioral patterns

- **Adaptive Modifications**: Apply layer-aware adaptive modifications to tensors based on their position and role in the model.

- **AI-Powered Recommendations**: Get suggestions for model improvements targeting specific capabilities like general performance, mathematical abilities, or reasoning capabilities.

## Components

The Tensor Slayer is built on several powerful modules:

1. **EnhancedTensorPatcher**: Provides methods for analyzing and modifying tensors with advanced operations.

2. **AITensorExplorer**: Uses AI to provide deeper insights into tensor patterns and model behaviors.

3. **SafetensorsExplorer**: Handles direct interaction with model files, especially in the safetensors format.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Tensor-Slayer.git
cd Tensor-Slayer

# Install dependencies
pip install -r requirements.txt
```
- The architecture uses Openrouter model (gemini-2.0) to reverse engineer the target LLM through LiteLLM implementation.
- Rename the .env.example to .env and update the OPENROUTER_API_KEY and OPENROUTER_MODEL with your credentials and desired model. I have been getting good results with Gemini 2.0 to reverse engineer the target LLMs.
  
## Dependencies

- Python 3.7+
- rich: For beautiful CLI interfaces
- typer: For command-line argument parsing
- safetensors: For safe tensor manipulation
- torch: Required for tensor operations
- numpy: For numerical operations
- matplotlib: For visualization

## Usage

### Quick Start

```bash
python model_explorer.py /path/to/your/model
```

### Interactive Commands

Once the explorer is running, you can use these commands:

- **explore**: Browse model structure and tensor types
- **investigate** \<query\>: Investigate specific model behaviors
- **analyze** \<tensor\>: Analyze detailed tensor statistics and patterns
- **patch**: Start guided patching session
- **adaptive** \<tensor\>: Apply layer-aware adaptive modifications
- **suggest**: Get AI-powered modification suggestions
- **help**: Show help information
- **exit**: Exit the explorer

### Example Workflows

#### Basic Model Exploration

```
Command> explore
```

This will show an overview of the model structure and highlight key tensor types like attention layers, MLP blocks, and embeddings.

#### Investigating Specific Behaviors

```
Command> investigate "why does the model avoid certain topics"
```

The tool will analyze relevant tensors and provide insights into model behaviors related to the query.

#### Analyzing and Modifying Tensors

```
Command> analyze model.layers.10.self_attn.q_proj.weight
Command> patch
```

The first command performs detailed analysis on a specific attention query projection weight tensor. The second command starts a guided session to modify this tensor.

#### Getting Improvement Suggestions

```
Command> suggest
```

This will provide AI-powered recommendations for tensor modifications that could improve specific model capabilities.

## Advanced Usage

### Adaptive Tensor Modification

```
Command> adaptive model.layers.5.mlp.down_proj.weight
```

This applies layer-aware modifications to the specified tensor based on its position and role in the model architecture.

### Batch Processing

For batch processing of multiple models, you can create a shell script:

```bash
#!/bin/bash
for model in /path/to/models/*; do
  python model_explorer.py "$model" --batch --operation scale --value 1.05 --output "$model_modified"
done
```

## Safety Considerations

- The tool creates backups before applying modifications
- Safety checks are performed before applying changes
- Preview functionality allows assessing impact before committing changes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project leverages various open-source libraries and research in the field of model interpretability and modification.
- Special thanks to the developers of the safetensors format and the Rich library for Python.

## Citation

If you use this tool in your research, please cite:

```
@software{tensor_slayer,
  author = {areu01or00},
  title = {Tensor Slayer},
  url = {https://github.com/areu01or00/tensor-slayer},
  year = {2025},
}
``` 
