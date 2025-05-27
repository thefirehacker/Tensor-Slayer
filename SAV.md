# System Evolution: Tensor Explorer to Web-Based Visual Editor (SAV - System Architectural Vision)

This document outlines the transformation of the Tensor Patching Agent project from a command-line tool into a web-based visual tensor explorer and editor.

**Abstract:** This System Architectural Vision (SAV) document details the journey of the Tensor Patching Agent from its origins as a Python-based command-line toolkit to a sophisticated, single-page web application for visual tensor exploration and editing. It covers the initial goals, key evolutionary steps, core design philosophy, current architectural vision including API specifications and UI componentry, and future development roadmap. The aim is to create a "hypothesis-driven model surgery toolkit" that allows users to visually inspect, analyze, and modify `.safetensors` model weights in an intuitive and powerful manner, leveraging both direct manipulation and AI-assisted investigation.

## 1. Initial State & Goals

*   **Initial Project:** A Python-based toolkit (`model_explorer.py`, `ai_tensor_explorer.py`, `enhanced_tensor_patcher.py`, `safetensors_explorer_cli.py`) for investigating and modifying `.safetensors` model weights, primarily through a command-line interface. It included AI-assisted investigation capabilities designed to help users understand tensor functionalities and relationships.
*   **User Goal:** Transform the project into:
    1.  A visually model editable framework, moving beyond the limitations of CLI for complex tensor manipulations and analysis.
    2.  A system capable of simple 2D/3D representation of specific tensor clusters (long-term) and enabling intuitive, potentially binary-level, editing of these tensors.
    3.  An environment that facilitates rapid hypothesis testing regarding model behavior and parameter impact.
*   **Initial Challenges Identified:**
    *   Limitations in the AI's ability to provide "informed results" due to tool constraints (e.g., the AI agent having limited context or simplified views of tensor data).
    *   Lack of a dedicated visualization component, making it difficult to grasp tensor structures and relationships intuitively.
    *   Complexity in managing the various CLI commands and their interactions, hindering a smooth workflow.
    *   Difficulty in directly observing the impact of changes without a visual feedback loop.

## 2. Evolution Steps & Decisions

### Step 2.1: Code Refinements (AI Tensor Explorer)

*   **Action:** Addressed some limitations in `ai_tensor_explorer.py` by:
    *   Refining tool definitions for the AI agent (e.g., `TensorValuesTool` was enhanced to support slicing, allowing the AI to request specific portions of tensors).
    *   Correcting tensor name access in `TokenEmbeddingComparisonTool` to ensure accurate data retrieval.
    *   Ensuring the `tensor_statistics` method was correctly placed within the `AITensorExplorer` class for proper functionality.
*   **Rationale:** To improve the underlying data access and analysis capabilities available to the AI agent. These refinements were crucial for providing more accurate and useful information, forming a stronger foundation for any future interface, including the web UI.

### Step 2.2: Evaluating Visualization Options

*   **Option 1: `MAV` (Model Activity Visualiser)**
    *   **Assessment:** `MAV` is a terminal-based tool for visualizing LLM activity *during generation*. While its plugin system for custom panels was promising for real-time insights, its dynamic nature (focused on live model runs) and terminal limitations (especially for complex 2D/3D visualization and interactive, fine-grained editing of static model weights) made it less ideal for the static tensor exploration and direct editing goals of this project.
    *   **Decision:** Decided against `MAV` to pursue a more flexible and controllable web-based UI better suited for inspecting and modifying quiescent model states.

*   **Option 2: Custom Web UI (FastAPI + HTML/JS)**
    *   **Assessment:** A localhost web UI offers significantly greater flexibility for custom visualizations (heatmaps, potential future 3D views), complex user interactions (drag-and-drop, direct cell editing), and seamless integration with the existing Python backend.
    *   **Decision:** Proceed with building a custom web UI. FastAPI was chosen for the backend due to its modern asynchronous capabilities, performance, and ease of use for creating RESTful APIs. Standard HTML, CSS, and JavaScript were selected for the frontend to ensure broad compatibility and control over the user experience.

### Step 2.3: Initial Web UI Scaffolding

*   **Backend (`main_web.py`):**
    *   **Action:** Created a FastAPI application.
    *   **Core Python Class:** Leveraged the `AITensorExplorer` class as the primary engine for model loading and tensor data access.
    *   **Functionality:**
        *   Initializes `AITensorExplorer` to load a model. The model path was initially hardcoded but quickly transitioned to be configurable via the `MODEL_PATH` environment variable for flexibility.
        *   Serves a static `index.html` page as the main entry point for the application.
        *   Provides an initial API endpoint (`/api/tensors`) to list all available tensors (name, shape, dtype) from the loaded model. This endpoint would later be expanded to include more metadata.
    *   **Rationale:** FastAPI provides a robust and efficient way to expose Python functionalities to a web frontend. Using `AITensorExplorer` ensures that the web UI benefits from the existing logic for handling `.safetensors` files and tensor data.

*   **Frontend (`templates/index.html`, `static/styles.css`, `static/scripts.js`):**
    *   **Action:** Created basic HTML structure for the page, CSS for initial styling, and JavaScript for fetching and displaying data.
    *   **Functionality:**
        *   Displays a simple list of tensors fetched from the `/api/tensors` endpoint.
        *   Included a basic search/filter bar for the tensor list.
        *   Placeholder sections were designated in the HTML for future detailed tensor information displays and editing controls.
    *   **Rationale:** To provide a foundational user interface that could be incrementally built upon, starting with the ability to see what tensors are in the model.

## 3. Core Philosophy: Hypothesis-Driven Exploration and Editing

Before diving deeper into implementation, it's crucial to establish the guiding philosophy for this tool. The goal is not just to display tensors, but to create an **interactive model surgery toolkit**. This involves:

*   **Internalizing the End Product:** We envision a system where users can:
    1.  Formulate hypotheses about model behavior (e.g., "Why does the model generate X specific token?", "How can I make the model less biased towards Y?", "What if I amplify/attenuate the weights in attention head Z of layer L?").
    2.  Use visualizations (like the global tensor canvas and detailed heatmaps) and analytical tools (like AI-driven investigation via the sidebar panel) to locate relevant tensors or patterns within tensors that might be responsible for or related to these hypotheses.
    3.  Visually inspect these tensors in an intuitive and aesthetically engaging way (the "Matrix-like" interface concept, with color-coding and clear structural representation).
    4.  Perform targeted edits directly on the visual representation of these tensors (e.g., clicking a cell in a heatmap and inputting a new value) or through clearly linked controls (e.g., applying a scaling factor to a selected slice).
    5.  Observe the impact of these edits by saving the modified model (or a patch file) and re-testing it in its intended environment.
*   **Reverse Coding the Vision:** This end-product vision dictates our development priorities. Each feature, from data fetching (e.g., `/api/tensor_slice`) to visualization rendering (Plotly.js heatmaps) to editing capabilities (hex-diff patching), is a step towards enabling this iterative loop of hypothesis, exploration, editing, and observation.
*   **Leveraging Existing Framework for the "HOW":**
    *   The existing `AITensorExplorer` and its associated tools (`TensorListTool`, `TensorStatisticsTool`, `TensorValuesTool`, `CodeAgent` for `investigate` commands) are fundamental to addressing the "HOW do we know what to edit?" question. These tools provide the analytical power, often AI-assisted, to form and test hypotheses about tensor functions.
    *   The web UI will act as a sophisticated frontend to these capabilities, making them more accessible and pairing them with direct visual feedback and manipulation tools. The "Investigation Panel" in the UI is a direct interface to this.
    *   `EnhancedTensorPatcher` remains the backend workhorse for applying the actual edits to the tensor data and saving new model versions or patch files.

## 3.1. User Experience (UX) Goals

The development is guided by the following UX goals:
*   **Intuitive Interaction:** Users should be able to navigate and interact with complex tensor data without a steep learning curve. Visual cues, clear labeling, and consistent interaction patterns are key.
*   **Information Density & Clarity:** The UI should present a large amount of information (e.g., overview of all tensors, detailed slice views) in a way that is digestible and not overwhelming. The global canvas and tabbed sidebar aim to achieve this.
*   **Direct Manipulation:** Where possible, users should be able to interact directly with visual representations of data (e.g., clicking on canvas thumbnails, selecting heatmap cells for editing).
*   **Responsiveness:** The application should feel fast and responsive, especially when loading data and rendering visualizations. Backend optimizations and efficient frontend rendering are crucial.
*   **Aesthetically Engaging:** While a technical tool, an aesthetically pleasing and "cool" interface (the "Matrix-like" or "tensor-slaying toolkit" vibe) can enhance user engagement and make the complex task of model surgery more enjoyable.
*   **Iterative Workflow:** The entire system is designed to support an iterative workflow of exploration, hypothesis, editing, and testing.

## 4. Next Steps (Now Largely Implemented or In Progress under "Architectural Evolution")

This section previously outlined the immediate tasks. Many of these have been incorporated into the broader architectural vision described in section 4.1.

*   **Backend - API for Sliced Tensor Data:**
    *   **Endpoint:** `/api/tensor_slice/{tensor_name}` (GET request).
    *   **Purpose:** To fetch a specific slice of a tensor for detailed visualization and potential editing.
    *   **Request Parameters:**
        *   `tensor_name`: Path parameter (URL-encoded).
        *   Query parameters for slice dimensions, e.g., `dim0_start=0&dim0_end=32&dim1_start=0&dim1_end=32` for a 2D slice of a 2D+ tensor. For higher-rank tensors, additional `dimN_idx=X` parameters would specify fixed indices for other dimensions not being sliced over for 2D display.
        *   If no slice parameters are provided for a high-dimensional tensor, the API might return a default slice (e.g., the first `[0:default_size, 0:default_size, ...]`) or an error/metadata indicating the full tensor is too large for direct transmission.
    *   **Response (JSON):**
        ```json
        {
          "tensor_name": "string",
          "original_shape": "[int, ...]", // Full shape of the tensor
          "slice_definition": "string", // e.g., "[0:32, 0:32]" or "{'dim0': (0,32), 'dim1': (0,32)}"
          "slice_shape": "[int, ...]",    // Shape of the returned data slice
          "dtype": "string",              // Data type of the tensor
          "data": "[[number, ...], ...]", // Nested list of numbers representing the slice values
          "min_value_in_slice": "number", // Minimum value in the returned slice
          "max_value_in_slice": "number"  // Maximum value in the returned slice
        }
        ```
    *   **Implementation:** Uses `AITensorExplorer` (and its underlying `SafetensorsExplorer`) to load the tensor, extract the specified slice, and return *all values* for that slice as JSON. This is crucial for detailed visualization in the frontend, moving beyond small generic samples.

*   **Frontend - Plotly.js Heatmap for 2D Slices:**
    *   **Integration:** Plotly.js integrated into `index.html` for rendering interactive charts.
    *   **Functionality (within "Focus View" tab in `static/scripts.js`):**
        *   When a tensor is selected (e.g., from the global canvas), and slice parameters are defined in the "Focus View", the frontend calls the `/api/tensor_slice/{tensor_name}` backend API.
        *   Uses Plotly.js to render this 2D slice data as an interactive heatmap in the `#tensor-visualization` div within the "Focus View".
        *   Implemented hover-over-cell functionality to display the exact value and indices.
        *   Implemented click-on-cell functionality to "select" a cell, displaying its value and full index, preparing for future direct editing. This selected cell's information can be used by the "Patch Editor Panel".

*   **Configuration:**
    *   The model path in `main_web.py` is configured via the `MODEL_PATH` environment variable (e.g., `model_path = os.getenv("MODEL_PATH", "default_model.safetensors")`). This allows users to easily point the application to different model files without code changes.
    *   Consideration for future environment variables: `HOST`, `PORT` for Uvicorn, `LOG_LEVEL`.

## 4.1. Architectural Evolution: Single-Page Monster Toolkit (Current Vision)

Based on further brainstorming and user feedback, the project has evolved towards a more integrated, single-page application, departing from the initial multi-column layout. This new vision aims for a "mega monster tensor slaying toolkit."

*   **Core UI Changes:**
    *   **Single-Page Application (SPA) Model:** All functionalities reside on a single HTML page (`index.html`), with dynamic content updates managed extensively by JavaScript (`static/scripts.js`).
    *   **Global Tensor Canvas (`#tensor-canvas-container`):**
        *   The central UI element is a zoomable/pannable canvas designed to display representations (e.g., compact thumbnails, mini heatmaps if feasible) of *all* or many tensors simultaneously. This replaces the simple list view, offering a spatial overview of the model.
        *   **Layout Strategy:** Tensors are arranged on this canvas using a defined X-Y axis mapping to provide a structured overview of the model architecture. The primary strategy is **Layer-Major, Type-Minor grouping**.
            *   *Layer-Major:* Tensors are grouped primarily by their inferred layer index.
            *   *Type-Minor:* Within each layer, tensors are sub-grouped or ordered by their general type (e.g., embeddings, attention components, MLP components, normalization weights).
            *   This layout aims to mirror common representations of neural network architectures, making it easier to locate specific components.
    *   **Integrated Toolkit Sidebar (`#toolkit-sidebar`):** A persistent sidebar houses various tools and views, organized into tabs for efficient space usage and focused interaction.
        *   **Focus View Tab:** Displays detailed information for a tensor selected from the canvas.
            *   **Content:** Tensor Name, Dtype, Full Shape, Number of elements.
            *   **Statistics:** Basic descriptive statistics (min, max, mean, std dev of the entire tensor – potentially fetched on demand or from `/api/tensors` metadata).
            *   **Slicing Controls (`#tensor-slicing-controls`):** Input fields for the user to define which 2D slice to view (especially for 3D+ tensors). For 2D tensors, it might default to the full tensor if dimensions are manageable, or a default slice (e.g., `[:64, :64]`). For higher-rank tensors, users specify start/end for two dimensions and fixed indices for others.
            *   **Action Button:** A "View/Refresh Slice" button triggers a call to `/api/tensor_slice/` with the current slice parameters.
            *   **Visualization Area (`#tensor-visualization`):** Renders the Plotly.js heatmap of the fetched tensor slice. Hover shows value/indices; click selects a cell, showing its value and full multi-dimensional index, making this information available for the Patch Editor.
        *   **Investigation Panel Tab:** For natural language queries to the backend AI (`AITensorExplorer`).
            *   **Input:** A textarea for users to type questions (e.g., "Which tensors are most important for X?").
            *   **Backend:** Queries are sent to a dedicated API endpoint (e.g., `/api/investigate`) which uses `AITensorExplorer`'s `investigate` method (powered by an LLM via LiteLLM and custom tools like `TensorListTool`, `TensorStatisticsTool`, `TensorValuesTool`).
            *   **Output:** Results (text, potentially formatted with tensor names that link back to the canvas/focus view) are displayed in this panel.
        *   **Patch Editor Panel Tab:** Manages and displays staged hex-diffs for tensor modifications before they are compiled into a patch file. (See Section 5.1 for hex-diff details).
            *   Lists pending changes (e.g., "tensor 'X', at offset Y, original_value Z, new_value W").
            *   Allows users to review, revert, or confirm individual changes.
        *   **Patch Manager Panel Tab:** Handles loading, saving, and annotating `.hexpatch` files.
            *   **Functionality:** Load an existing patch file to apply to the current model, save current staged changes as a new patch file, list available patches, allow for textual annotations (e.g., "Patch to enhance French translation performance").
    *   **Header (`.toolkit-header`):** Global controls (e.g., Load Model (via file input or new `MODEL_PATH` entry), Save All Staged Patches, Export Patched Model, Settings) and model status display (`#model-status-header`).
    *   **Footer (`.toolkit-footer`):** A quick query/filter bar for the tensor canvas (e.g., text filter for tensor names on the canvas).

*   **Backend Implications for Canvas Layout (`/api/tensors` Enhancement):**
    *   The `/api/tensors` endpoint (GET request) is enhanced to provide rich metadata for each tensor, crucial for frontend rendering and positioning on the X-Y canvas.
    *   **Response Structure:** Returns a JSON list, where each item is an object representing a tensor:
        ```json
        [
          {
            "name": "string", // Full tensor name
            "shape": "[int, ...]",
            "dtype": "string",
            "numel": "int", // Total number of elements
            "metadata_for_canvas": {
              "layer_index": "int", // Inferred layer index, -1 for non-layer-specific (e.g., embeddings)
              "type_group": "string", // e.g., "embedding", "attention_qkv", "attention_out", "mlp_gateup", "mlp_down", "norm", "output_head"
              "specific_type_key": "string", // Fine-grained key for styling, e.g., "model.layers.0.self_attn.q_proj"
              "canvas_x": "float", // Calculated X coordinate for canvas placement
              "canvas_y": "float", // Calculated Y coordinate for canvas placement
              // Basic stats could also be pre-calculated and included here if lightweight:
              // "min_val": "number", "max_val": "number", "mean_val": "number"
            }
          },
          // ... more tensor objects
        ]
        ```
    *   **Rule-Based Parsing for Metadata:** The backend performs rule-based parsing of tensor names to infer `layer_index` and `type_group`/`specific_type_key`.
        *   **Layer Index Extraction:** Typically uses regular expressions to find patterns like `layers.X.`, `block.X.`, etc., where `X` is the layer number. Embeddings might get layer `0` or `-1`, final normalization/output heads get a high layer number.
        *   **Type Grouping Examples:**
            *   `embed_tokens`, `wte`, `wpe`: `type_group: "embedding"`
            *   `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`: `type_group: "attention_qkv"`
            *   `self_attn.o_proj`: `type_group: "attention_out"`
            *   `mlp.gate_proj`, `mlp.up_proj`: `type_group: "mlp_gateup"`
            *   `mlp.down_proj`: `type_group: "mlp_down"`
            *   `input_layernorm`, `post_attention_layernorm`, `model.norm`: `type_group: "norm"`
            *   `lm_head`, `output`: `type_group: "output_head"`
        *   `specific_type_key` is often the part of the tensor name that uniquely identifies its role within a block or layer, used for fine-grained CSS styling.
    *   **Future Enhancement:** Explore using an LLM (potentially via LiteLLM, as currently used for other AI features) to assist in generating this layout metadata, especially for unfamiliar model architectures. This LLM-generated layout could be a one-time process per model or a configurable option, offering more adaptive or nuanced layouts.

*   **Frontend JavaScript Overhaul (`static/scripts.js`):**
    *   Significant refactoring to manage the SPA state.
    *   Implement the tabbed interface for the toolkit sidebar, dynamically showing/hiding content.
    *   Dynamically render tensor thumbnails on the global canvas based on data from `/api/tensors`. This includes setting `data-` attributes (like `data-specific-type` from `specific_type_key`) on thumbnail elements for CSS styling.
    *   Handle all user interactions: canvas zoom/pan, tensor thumbnail selection, sidebar tab switching, API calls for slice data, Plotly rendering, interactions with investigation panel and patch panels.
    *   Manage application state (e.g., currently selected tensor, current slice definition, staged patches).

*   **Styling (`static/styles.css`):**
    *   `static/styles.css` has been updated to provide the basic structure for this new single-page layout (header, main content area with canvas and sidebar, footer).
    *   Crucially, it includes specific CSS rules to style tensor thumbnails on the canvas. These rules use attribute selectors based on the `data-specific-type` (derived from `specific_type_key` in the backend metadata) and `data-type-group` attributes set on the thumbnail elements.
    *   **Example Styling Rules (Conceptual, actual in `styles.css`):**
        *   `.tensor-thumbnail[data-specific-type*="embed"] { background-color: #e6fffa; border-left: 3px solid #00bfa5; }`
        *   `.tensor-thumbnail[data-specific-type*="layernorm"] { background-color: #fff0f5; border-left: 3px solid #db7093; }`
        *   `.tensor-thumbnail[data-specific-type*="self_attn.q_proj"] { background-color: #e0f7fa; }`
        *   `.tensor-thumbnail.selected { border-color: #ff4500; transform: scale(1.05); }`
    *   This visual differentiation is key to quickly identifying tensor types on the canvas.

This architectural shift prioritizes information density, direct manipulation, and a tool-rich environment, aligning with the "tensor slaying toolkit" concept and the "reverse coding the end product" philosophy. The goal is a powerful, yet intuitive, platform for deep model introspection and modification.

## 5. Future Goals (Post-Current Focus - some may be integrated earlier)

*   **Visualization Enhancements:**
    *   Beyond heatmaps: distribution plots (histograms) for tensor values (full tensor or slices) within the "Focus View".
    *   Small, indicative visualizations directly on the tensor thumbnails on the global canvas (e.g., micro-histograms or sparkline-style value ranges), if performance permits.
    *   Explore 3D visualizations for rank-3 tensor slices if a suitable lightweight library and interaction model can be found.
*   **Editing Functionality - Deeper Integration:**
    *   **Direct Heatmap Editing:** Allow users to click a cell in the Plotly.js heatmap, input a new value, and have this change staged in the "Patch Editor Panel".
    *   **Slice Operations:** UI elements in "Focus View" to apply operations to a selected slice (e.g., scale by X, add Y, clamp values, zero out). These operations would also generate corresponding hex-diffs.
    *   **Backend API for Edits:** A robust `POST /api/edit_tensor` endpoint that takes detailed edit information (tensor name, offset/indices, old value, new value, or operation type) and uses `EnhancedTensorPatcher` to stage these changes. This is closely tied to the Hex-Diff Patching workflow (Section 5.1).
    *   Real-time (or near real-time) feedback in the UI about the success/failure of edit staging operations.
    *   Mechanism to optionally "refresh" or re-fetch tensor slice data after edits are staged to see their immediate effect in the heatmap (though the canonical data remains the base model until a patch is applied and the model reloaded/exported).
*   **Binary Editing Representation:**
    *   In the "Focus View" or a dedicated "Low-Level View" tab, provide an option to see tensor slice data as raw hexadecimal (or binary) values.
    *   Allow direct modification of these hex values, with changes translating into the hex-diff patching system. This is for advanced users or specific use cases requiring byte-level precision.
*   **Tensor Clustering and Relationship Visualization:**
    *   Investigate methods to compute and display 2D/3D representations of tensor *clusters* based on similarity (e.g., activation patterns if ever integrated, or weight similarity). This might involve backend processing using dimensionality reduction techniques like PCA or t-SNE, with results shown on a separate canvas or overlay.
    *   Visualize relationships (e.g., learned or attention-based) between tensors.

## 5.1. Refined Patching Workflow: Hex-Diff Patch Files

This is a critical component for enabling non-destructive, shareable, and auditable model modifications.

*   **Concept:** Instead of saving the entire modified model file after each minor edit or session of edits, the toolkit will generate and manage specialized "hex-diff patch files" (e.g., with a `.hexpatch` extension). These files contain only the changes made to the base model.
*   **Mechanism:**
    1.  **Tracking Modifications:** As the user makes edits (e.g., changing specific values in a tensor slice via heatmap interaction, applying scaling operations, or direct hex editing), these modifications are captured by the frontend and sent to the backend. The backend, likely via an augmented `EnhancedTensorPatcher`, translates these logical edits into precise memory offsets and byte-level changes within the target tensor.
    2.  **Staging Changes:** Edits are initially "staged" (e.g., in the "Patch Editor Panel"). Each staged change would internally represent:
        *   `tensor_name`: The name of the tensor being modified.
        *   `offset_in_tensor_bytes`: The byte offset within that tensor's data stream where the change begins.
        *   `original_hex_sequence`: The sequence of hexadecimal characters representing the original bytes at that location (for verification and reversibility).
        *   `new_hex_sequence`: The sequence of hexadecimal characters representing the new bytes.
        *   (Optionally) `dtype` and `shape_at_edit_time` for context.
    3.  **Compiling the Patch File:** When the user chooses to save their work (e.g., "Save Patches"), the system compiles all staged changes into a structured patch file. This file would typically be JSON or a similar human-readable format, containing a list of these diff entries.
        ```json
        // Example .hexpatch file structure
        {
          "base_model_checksum": "sha256_hash_of_original_model_for_validation", // Optional but recommended
          "description": "User-provided description of the patch's purpose",
          "patches": [
            {
              "tensor_name": "model.layers.10.self_attn.v_proj.weight",
              "offset_bytes": 1024,
              "original_hex": "0A1B2C3D",
              "new_hex": "FFAABBCC"
            },
            // ... more patch entries
          ]
        }
        ```
    4.  **Applying Patches:** The toolkit will be able to load a `.hexpatch` file. When loaded, it can apply these hex modifications to the *original base model* to reconstruct the patched model state in memory or for export.
*   **Advantages:**
    *   **Efficiency:** Patch files are significantly smaller than full model snapshots, especially for sparse edits affecting only a few tensors or parts of tensors. This saves disk space and makes sharing easier.
    *   **Transparency & Auditability:** The patch file provides a clear, human-readable (for the hex parts) or machine-parsable record of exactly what was changed, where, and from what to what. This is invaluable for debugging, collaboration, and understanding the evolution of modifications.
    *   **Modularity & Composability:** Patches could potentially be combined, selectively applied, or version-controlled independently of the base model. (Composability needs careful handling of overlapping patches).
    *   **Reduced I/O:** Less disk writing during the interactive editing session. Edits are primarily in memory or staged until a "Save Patch File" or "Export Patched Model" action.
*   **Implementation Consideration:** This requires a robust backend mechanism (extending `EnhancedTensorPatcher` or a new `HexPatcher` class):
    *   Precisely map UI-level edits (e.g., change value at `[idx1, idx2]` in a float32 tensor) to byte offsets and hex sequences, considering the tensor's `dtype` and memory layout (e.g., row-major/column-major, endianness).
    *   Generate the hex diffs by reading original bytes before modification.
    *   Reliably apply these hex diffs to a base model's byte stream to materialize a fully patched model when needed (e.g., for final export or for loading a patched state for further editing).
    *   Handle potential conflicts if multiple patches modify the same byte locations (e.g., last patch wins, or raise error).

## 5.2. Advanced Model Surgery: Tensor/Module Transplantation (Future Goal)

Building upon the fine-grained editing capabilities, a highly impactful future direction is to enable the transplantation of entire tensors or even modules (collections of related tensors, e.g., a full attention block or MLP) from a source model to a target model. This moves beyond value editing into architectural recombination.

*   **Concept:** Users could select a tensor (e.g., `transformer.wte.weight`) or a defined module (e.g., `model.layers.10.self_attn`) in a target model and replace it with a corresponding tensor/module from a specified source model.
*   **Potential Use Cases:**
    *   **Knowledge Transfer:** Importing pre-trained embeddings or specialized layers.
    *   **Architectural Experimentation:** Testing different component versions (e.g., swapping attention mechanisms).
    *   **Modular Upgrades:** Replacing specific parts of a model with improved versions.
*   **Key Challenges & Considerations:**
    *   **Tensor/Module Mapping:**
        *   *Naming Conventions:* Different models use different tensor naming schemes. A robust mapping mechanism (manual, rule-based, or AI-assisted) would be essential to identify corresponding components.
        *   *Structural Equivalence:* Defining what constitutes a "module" and ensuring the source and target modules have compatible roles and interfaces.
    *   **Architectural Compatibility & Connectivity:**
        *   The dimensions of the transplanted tensor/module must align with the tensors it connects to in the target model (e.g., `in_features` of a weight must match `out_features` of the preceding layer).
        *   Changes in fundamental properties like vocabulary size (if swapping token embeddings) would necessitate corresponding changes in related tensors (e.g., the language model head's output layer).
    *   **Shape and Size Mismatches:**
        *   If a transplanted tensor has a different shape/size than the original, this is a major challenge.
        *   Simply patching bytes is insufficient. The `.safetensors` file structure itself would need modification:
            *   The metadata (header) for the modified tensor (shape, dtype, offset) must be updated.
            *   Critically, the data offsets for *all subsequent tensors* in the file would need to be recalculated and updated in the header.
            *   The overall file size would change.
        *   This implies a re-serialization of the model from the point of modification onwards, rather than an in-place patch.
    *   **Data Type (`dtype`) Conversion:** Handling or warning about `dtype` mismatches between source and target tensors.
    *   **Patching Mechanism:**
        *   The `.hexpatch` system is designed for in-place, same-size modifications.
        *   Tensor transplantation, especially with size changes, would require a new type of "patch" or operation. This might involve storing the entire new tensor data within the patch and detailed instructions for re-calculating subsequent tensor offsets and rebuilding the model's header.
        *   Alternatively, the operation might directly result in saving a *new, fully modified model file* rather than a patch.
*   **Phased Implementation Approach:**
    1.  **Phase 1 (Simplest):** Support replacement of individual tensors with identical names, shapes, and dtypes from a source model. The "patch" could still be a large hex-diff in this case, or a special "replace_tensor_data" instruction within an evolved patch file format.
    2.  **Phase 2 (Mapping):** Allow user-defined mapping between source and target tensor names, still requiring identical shapes and dtypes.
    3.  **Phase 3 (Handling Size/Shape Differences - MVP):**
        *   **Core User Story:** The user wants to replace *any* tensor in a target model with *any* tensor from a source model, regardless of shape/size mismatches, and save the result as a *new model file*. This acknowledges that finding identically shaped tensors is often not feasible for experimental purposes.
        *   **Implementation - New Tool Page:** Introduce a new dedicated page/view in the toolkit (e.g., "Tensor Transplanter" or "Model Mixer").
            *   Allows loading a "Source Model" and a "Target Model" (via file upload or path input).
            *   Displays selectable lists of tensors (name, shape, dtype) for both models.
            *   The user selects one tensor from the source and one tensor in the target model to be replaced.
            *   An input field for the desired output path/name for the new model.
            *   A "Proceed" button triggers the backend operation.
        *   **Backend Operation:** The backend will load the source tensor's data and metadata. It will then load the target model's structure. A *new model* is constructed in memory by taking all tensors from the target model, replacing the specified target tensor's data and metadata (shape, dtype) with those from the source tensor. If the new tensor's byte size differs from the original, this necessitates a full re-serialization of the model: all subsequent tensor data offsets in the new model's header must be recalculated. The backend then saves this entirely new model structure to the specified output path. This explicitly creates a new file, not a patch.
        *   **Crucial User Warning:** The UI *must* prominently warn the user that while the tool can perform this swap, if the transplanted tensor's shape is not compatible with its connecting layers in the target model's architecture (e.g., `output_features` of tensor A do not match `input_features` of subsequent tensor B), the resulting model will likely be architecturally unsound and **will fail during inference** (e.g., with shape mismatch errors). The tool facilitates the raw structural change; the user is responsible for the architectural validity and downstream consequences of such arbitrary swaps. This is key to managing expectations for the "replace any shape with any shape" capability.
        *   AI-assistance could *later* be explored to analyze architectural compatibility or suggest concomitant changes, but the MVP allows the direct swap first.
*   **UI/Backend Implications (Recap for MVP):**
    *   A dedicated "Tensor Transplanter" or "Model Mixer" interface.
    *   Capabilities to load multiple models (source and target).
    *   Visual tools for selecting and mapping tensors/modules.
    *   Clear communication of compatibility issues and the impact of structural changes.
    *   Backend logic to handle the re-serialization of `.safetensors` files if sizes change.

This feature would significantly elevate the toolkit's capabilities, turning it into a sophisticated platform for deep architectural experimentation and modular model construction.


## 6. Technologies Used

*   **Backend:**
    *   Python 3.x
    *   FastAPI: For the web server and API endpoints.
    *   Uvicorn: ASGI server to run FastAPI.
    *   Safetensors: Library for loading `.safetensors` model files.
    *   NumPy: For numerical operations on tensor data.
    *   LiteLLM: For proxying requests to various LLMs (for AI-assisted investigation features).
    *   Standard Python libraries (os, json, etc.).
*   **Frontend:**
    *   HTML5
    *   CSS3
    *   JavaScript (ES6+)
    *   Plotly.js: For interactive charts and heatmaps.
*   **Core Logic Classes (Python):**
    *   `AITensorExplorer`: Orchestrates model loading, data access, and AI-assisted investigation.
    *   `SafetensorsExplorer`: Low-level interaction with `.safetensors` files.
    *   `EnhancedTensorPatcher`: Handles the application of modifications to tensor data and saving models/patches. Will be augmented for hex-diff capabilities.
*   **Development & Tooling:**
    *   Git: For version control.
    *   Environment variables: For configuration (e.g., `MODEL_PATH`).

## 7. Conclusion

The evolution of the Tensor Patching Agent into a web-based visual editor represents a significant step towards creating a powerful and intuitive "model surgery" toolkit. By combining a rich visual interface with hypothesis-driven exploration, direct manipulation capabilities, AI-assisted analysis, and a robust patching mechanism, the project aims to empower researchers and developers to understand, debug, and refine neural network models with unprecedented granularity and control. The architectural vision outlined in this document provides a roadmap for achieving this ambitious goal, emphasizing flexibility, user experience, and a strong connection between visual feedback and underlying model data.

---
This document will be updated as the project evolves. 