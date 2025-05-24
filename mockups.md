# Tensor Slaying Toolkit - UI Mockups (ASCII Art)

This document presents a series of ASCII art mockups to visualize the "Tensor Slaying Toolkit."
The goal is to conceptualize the user interface and interactions for advanced model analysis and editing.

## Mockup 1: Main Dashboard & Conversational Input

```
+---------------------------------------------------------------------------------------------------+
| Tensor Slaying Toolkit                                               Model: Qwen_0.6B (Editable)  |
+---------------------------------------------------------------------------------------------------+
| [ Global Actions: Load Model | Save Patches | Settings | Help ]                                  |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Tensor List (Scrollable)              |      | Conversation / Investigation Log             | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | - model.embed_tokens.weight         |^|      | > User: Why is the model avoiding topics ... | |
|   | - model.layers.0.self_attn.q_proj   | |      | > Sys: Analyzing query...                    | |
|   | - model.layers.0.self_attn.k_proj   | |      |                                              | |
|   | - model.layers.0.self_attn.v_proj   |#|      |                                              | |
|   | - model.layers.0.self_attn.o_proj   |#|      |                                              | |
|   | - model.layers.0.mlp.gate_proj      |#|      |                                              | |
|   | ... (many more tensors) ...         |v|      |                                              | |
|   +---------------------------------------+      +----------------------------------------------+ |
|                                                                                                   |
|   +---------------------------------------------------------------------------------------------+ |
|   | [ Converse with Model / Tensors:                                                            | |
|   | [_________________________________________________________________________________________] | |
|   | [Submit Query]                                                                              | |
|   +---------------------------------------------------------------------------------------------+ |
|                                                                                                   |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Visualization (Heatmap/Graph)         |      | Tensor Slaying Toolkit / Actions             | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | (Select a tensor or run investigation |      | (Contextual tools appear here)               | |
|   |  to populate this area)               |      |                                              | |
|   |                                       |      |                                              | |
|   |                                       |      |                                              | |
|   |                                       |      |                                              | |
|   |                                       |      |                                              | |
|   +---------------------------------------+      +----------------------------------------------+ |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 2: Conversational Query Input

(Focus on the conversation input part of Mockup 1)
```
+---------------------------------------------------------------------------------------------+
| [ Converse with Model / Tensors:                                                            |
| [ Why is this model censoring information related to the CCP and Tiananmen Square?        ] |
| [Submit Query]                                                                              |
+---------------------------------------------------------------------------------------------+
```

## Mockup 3: Investigation Response & Candidate Tensors

```
+---------------------------------------------------------------------------------------------------+
| ... (Header and Tensor List as before) ...                                                        |
+---------------------------------------------------------------------------------------------------+
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Tensor List (Scrollable)              |      | Conversation / Investigation Log             | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | - model.embed_tokens.weight         |^|      | > User: Why CCP censorship?                  | |
|   | - model.layers.0.self_attn.q_proj   | |      | > Sys: Analyzing potential censorship vectors| |
|   |   ...                               | |      |   Identified patterns in activation/weights. | |
|   | > model.layers.15.mlp.gate_proj (*) |#|      |   Candidate Tensors for review:              | |
|   | > model.layers.22.self_attn.o_proj(*)|#|      |   1. [model.layers.15.mlp.gate_proj]       | |
|   | - lm_head.weight                    |#|      |   2. [model.layers.22.self_attn.o_proj]    | |
|   | ... (many more tensors) ...         |v|      |   3. [model.embed_tokens.weight] (subset)  | |
|   +---------------------------------------+      |   Click tensor to visualize and edit.        | |
|                                                +----------------------------------------------+ |
| ... (Visualization and Toolkit panels update based on selection) ...                              |
+---------------------------------------------------------------------------------------------------+
```
(*) indicates highlighted/suggested tensors.

## Mockup 4: Deep Dive - Tensor Visualization & Contextual Toolkit

(Assuming `model.layers.15.mlp.gate_proj` is selected)
```
+---------------------------------------------------------------------------------------------------+
| ... (Header, Tensor List, Conversation Log as before, tensor selected) ...                        |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Visualization: mlp.gate_proj (Slice)  |      | Tensor Slaying Toolkit: mlp.gate_proj        | |
|   |---------------------------------------|      |----------------------------------------------| |
|   |   Dim0: [ 0 ] to [ 31 ] View Slice    |      | [ General Info: Shape, DType, Stats ]        | |
|   |   Dim1: [ 0 ] to [ 31 ]               |      |                                              | |
|   | +-----------------------------------+ |      | [ Analyze Selection (requires selection) ]     | |
|   | | Heatmap Display                   | |      | [ Highlight Values: [ < 0.01 ] [ > 0.9 ] ]   | |
|   | | ############**####                | |      |                                              | |
|   | | ###**#######**##                | |      | [ Operations: ]                              | |
|   | | **##########**####                | |      |   - [ View/Edit as Binary/Hex ]              | |
|   | | ####**######**####                | |      |   - [ Nullify/Zero Out ]                     | |
|   | | (Colors indicate values)          | |      |   - [ Scale Values ]                         | |
|   | +-----------------------------------+ |      |   - [ Apply Custom Function... ]             | |
|   |                                       |      |   - [ Invert Values (Selected) ]             | |
|   +---------------------------------------+      +----------------------------------------------+ |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 5: Binary/Hex Editing Suite

(Activated from Toolkit in Mockup 4)
```
+---------------------------------------------------------------------------------------------------+
| ...                                                                                               |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Visualization: mlp.gate_proj (Slice)  |      | Binary/Hex Editor: mlp.gate_proj (Slice)     | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | +-----------------------------------+ |      |   View Mode: [ Hex ] [ Binary ] [ Float ]    | |
|   | | Heatmap (Selection Highlighted)   | |      |   Coords: (row,col) / (idx) Value            | |
|   | | ##########[**]####                | |      |   (0,12): 3f8a12be (Hex) / 0.0673 (F32)      | |
|   | | ###[**]#####[**]##                | |      |   (0,13): [bf01ccde] <--- Edit Here          | |
|   | | [**]########[**]####                | |      |   (1,3): 00111010... (Bin)                   | |
|   | +-----------------------------------+ |      |   ...                                        | |
|   |                                       |      |   [Apply Changes to Staging] [Cancel]        | |
|   +---------------------------------------+      +----------------------------------------------+ |
| ...                                                                                               |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 6: Visual Selection & Analysis

```
+---------------------------------------------------------------------------------------------------+
| ...                                                                                               |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Visualization: mlp.gate_proj (Slice)  |      | Tensor Slaying Toolkit: mlp.gate_proj        | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | Heatmap with user-dragged selection:  |      | [ General Info ]                             | |
|   | +-----------------------------------+ |      |                                              | |
|   | | ########xxxx####                | |      | [ Analyze Selection: ]                       | |
|   | | ###xxxxxxx**##                | |      |   - Values selected: 6                         | |
|   | | xx**######**####                | |      |   - Min: -0.5, Max: 0.8, Avg: 0.1            | |
|   | | (x = selected region)           | |      |   - [ View Values of Selection ]             | |
|   | +-----------------------------------+ |      |   - [ Correlate with Vocab... ]              | |
|   +---------------------------------------+      +----------------------------------------------+ |
| ...                                                                                               |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 7: Targeted Tensor Operations on Selection

(Following Mockup 6, "Operations" expanded)
```
+---------------------------------------------------------------------------------------------------+
| ...                                                                                               |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Visualization (Selection Persists)    |      | Tensor Slaying Toolkit: Operations           | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | ... Heatmap ...                       |      | Selected Region (6 values):                  | |
|   |                                       |      |   [ Nullify Selection (Set to 0) ]           | |
|   |                                       |      |   [ Scale Selection by: [ 0.5_ ] ] [Go]      | |
|   |                                       |      |   [ Add to Selection:  [ 0.1_ ] ] [Go]      | |
|   |                                       |      |   [ Set Selection to:  [ ____ ] ] [Go]      | |
|   |                                       |      |   [ Apply Custom PyFn to Selection: [fn.py] ]| |
|   |                                       |      |   [ Stage Operation ]                        | |
|   +---------------------------------------+      +----------------------------------------------+ |
| ...                                                                                               |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 8: Hypothesis-Driven Patching & Staging

```
+---------------------------------------------------------------------------------------------------+
| ...                                                                                               |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | Staged Patches / Hypothesis Testing   |      | Conversation / Test Results                  | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | 1. mlp.gate_proj (Slice 0:5,10:15)   |      | > Sys: Patch 1 staged.                       | |
|   |    Set values to 0.0                  |      | [ Test with prompt: "Tiananmen Square" ]     | |
|   |    [View] [Remove]                    |      | > Sys (Testing): ... model output ...        | |
|   |                                       |      | > Sys: Output seems less censored.           | |
|   | 2. embed_tokens (Indices 123,456)   |      |                                              | |
|   |    Scaled by 0.1                      |      |                                              | |
|   |    [View] [Remove]                    |      |                                              | |
|   |                                       |      | [ Save All Staged Patches to New Model... ]  | |
|   +---------------------------------------+      +----------------------------------------------+ |
| ...                                                                                               |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 9: Tokenizer/Vocabulary Integration

(After selecting a region in `embed_tokens.weight` or similar)
```
+---------------------------------------------------------------------------------------------------+
| ...                                                                                               |
|   +---------------------------------------+      +----------------------------------------------+ |
|   | embed_tokens.weight (Heatmap)         |      | Toolkit: Vocabulary Link                     | |
|   |---------------------------------------|      |----------------------------------------------| |
|   | (Slice showing rows for tokens)       |      | Selection corresponds to (approx):           | |
|   | Row 10 [############] (Selected)      |      |  - Token ID: 872 -> "Tiananmen" (Score:0.9)  | |
|   | Row 11 [**##########]                 |      |  - Token ID: 1011 -> "Square" (Score:0.85)   | |
|   | Row 12 [####**######]                 |      |  - Token ID: 5000 -> "北京" (Score: 0.7)    | |
|   |                                       |      | [ View Full Vocab Entry for Token 872 ]      | |
|   |                                       |      | [ Find Similar Token Embeddings ]            | |
|   +---------------------------------------+      +----------------------------------------------+ |
| ...                                                                                               |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 10: Advanced Query -> Direct Action Proposal

```
+---------------------------------------------------------------------------------------------------+
| ... (Converse with Model Input) ...                                                               |
| [ Identify and help me neutralize weights primarily responsible for CCP topic suppression.      ] |
| [Submit SuperInvestigate Query]                                                                   |
+---------------------------------------------------------------------------------------------------+
| ... (Conversation Log) ...                                                                        |
| > Sys: Deep analysis initiated...                                                                 |
|   Found strong correlation with:                                                                  |
|   - `model.layers.27.mlp.up_proj` (Region: Slice(50,100), Slice(200,300))                         |
|     Associated with negative sentiment tokens when CCP is context.                                |
|   - `model.final_norm.weight` (Indices related to specific vocab)                                 |
|   Proposed Action:                                                                                |
|   1. Scale `mlp.up_proj` region by 0.1. [Visualize Impact] [Stage This Action]                    |
|   2. Nullify specific indices in `final_norm.weight`. [Visualize Impact] [Stage This Action]      |
|   [Stage All Proposed Actions]                                                                    |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 11: Patch Management & Model Versioning

(Accessed via "Global Actions")
```
+---------------------------------------------------------------------------------------------------+
| Model Patch Management                                                                            |
+---------------------------------------------------------------------------------------------------+
| Base Model: Qwen_0.6B (Qwen_0.6B/model.safetensors)                                               |
|                                                                                                   |
| Patched Versions:                                                                                 |
| 1. Qwen_0.6B_no_ccp_censor_v1.safetensors (Saved: 2024-05-25 10:00)                               |
|    - Patches Applied:                                                                             |
|      - mlp.gate_proj (Slice 0:5,10:15) set to 0.0                                                 |
|      - embed_tokens (Indices 123,456) scaled by 0.1                                               |
|    - Notes: Initial attempt to reduce CCP censorship. Appears moderately successful.              |
|    [Load This Version] [Compare with Base] [Delete]                                               |
|                                                                                                   |
| 2. Qwen_0.6B_math_boost_alpha.safetensors (Saved: 2024-05-24 15:30)                               |
|    - Patches Applied: ...                                                                         |
|    - Notes: ...                                                                                   |
|    [Load This Version] [Compare with Base] [Delete]                                               |
|                                                                                                   |
| [ Create New Patch Branch from Base ]                                                             |
+---------------------------------------------------------------------------------------------------+
```

## Mockup 12: Settings/Configuration Panel

```
+---------------------------------------------------------------------------------------------------+
| Settings                                                                                          |
+---------------------------------------------------------------------------------------------------+
| Model Management:                                                                                 |
|  - Default Model Path: [/path/to/models/Qwen_0.6B_________________] [Browse]                     |
|  - Auto-load last model on startup: [X] Yes [ ] No                                                |
|                                                                                                   |
| Visualization:                                                                                    |
|  - Default Heatmap Colorscale: [ Viridis ] (Dropdown: Plasma, Jet, etc.)                          |
|  - Max 2D Slice Size for Auto-Load: [ 128 ] x [ 128 ]                                             |
|                                                                                                   |
| Tensor Slaying Toolkit:                                                                           |
|  - Default Binary Edit Mode: [ Hex ] (Dropdown: Binary, Float32)                                  |
|  - Custom Python Functions Path: [/path/to/my_tensor_functions/_____] [Browse]                  |
|                                                                                                   |
| [Save Settings] [Restore Defaults]                                                                |
+---------------------------------------------------------------------------------------------------+ 