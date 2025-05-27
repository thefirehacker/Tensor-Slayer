# Tensor Slayer

## Manipulate target LLM's weights and tensors through binray file editing to improve performance and achieve inferenceless mechanistic interpretability

Using LLMs and Agentic framework to hex patch weights and tensors in order to improve LLMs without Finetuning/datasets.
Further, added a novel approach to perform static analysis on binary files of the models for better interpretability - this kind of interpretability is not meant to be a replacement of inference-based interpretability but to rather act as complementary.

## Extensively tested on Qwen_0.6B and Deepseek_r1_1.5B, indicating massive relative peformance boosts. Check reults folder for the patches and eval results.

![image](https://github.com/user-attachments/assets/84984a63-786d-42bb-8851-f910e61e6f59)

## Model Patching

### Model Enhancements
- Powerful LLMs can be equipped with agentic workflow to analyse, modify and improve target LLM by manipulating - Attention, MLP, Input, Output Layers of the target LLM by hex patching model binaries.


### Direct Tensor Patching
By focusing on the precise tensors that influence specific behaviors, enable targeted modifications that can yield significant performance improvements without the overhead of fine-tuning.

### Static Binary Analysis
Novel approach allows you to understand model structure and behavior patterns by analyzing the binary weights directly, providing insights normally only available through extensive inference testing.


### Results 

Tested extensively on Qwen_0.6B model. Leading to substantial relative performance gain on both MMLU and HumanEval Benchmarks.

![image](https://github.com/user-attachments/assets/3a1befc0-b08b-4847-871f-327394708166)

![image](https://github.com/user-attachments/assets/aa1a8d6f-4301-42fb-9bac-475e1482df3a)

![image](https://github.com/user-attachments/assets/5f4a5f1e-9cf7-4b64-b6e0-09a14b64112b)

![image](https://github.com/user-attachments/assets/fcd09a8f-ab2a-4609-a2a6-b1c21b4a188d)

![image](https://github.com/user-attachments/assets/db3cf4b0-1a70-4476-bd98-58428946bf91)

![image](https://github.com/user-attachments/assets/90ee8795-8618-47ba-8e87-73d6cbbb2257)

![image](https://github.com/user-attachments/assets/c8959cc5-d612-4216-a5e4-87d1e63d09ab)



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
git clone https://github.com/yourusername/tensor-slayer.git
cd tensor-slayer

# Install dependencies
pip install -r requirements.txt
```

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

If you use this tool in your research, please cite.

