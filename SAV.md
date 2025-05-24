# System Evolution: Tensor Explorer to Web-Based Visual Editor (SAV - System Architectural Vision)

This document outlines the transformation of the Tensor Patching Agent project from a command-line tool into a web-based visual tensor explorer and editor.

## 1. Initial State & Goals

*   **Initial Project:** A Python-based toolkit (`model_explorer.py`, `ai_tensor_explorer.py`, `enhanced_tensor_patcher.py`, `safetensors_explorer_cli.py`) for investigating and modifying `.safetensors` model weights, primarily through a command-line interface. It included AI-assisted investigation capabilities.
*   **User Goal:** Transform the project into:
    1.  A visually model editable framework.
    2.  A system capable of simple 2D/3D representation of specific tensor clusters and enabling binary editing of these tensors.
*   **Initial Challenges Identified:**
    *   Limitations in the AI's ability to provide "informed results" due to tool constraints.
    *   Lack of a dedicated visualization component.
    *   Complexity in managing the various CLI commands and their interactions.

## 2. Evolution Steps & Decisions

### Step 2.1: Code Refinements (AI Tensor Explorer)

*   **Action:** Addressed some limitations in `ai_tensor_explorer.py` by:
    *   Refining tool definitions (e.g., `TensorValuesTool` to support slicing).
    *   Correcting tensor name access in `TokenEmbeddingComparisonTool`.
    *   Ensuring the `tensor_statistics` method was correctly placed within the `AITensorExplorer` class.
*   **Rationale:** To improve the underlying data access and analysis capabilities available to the AI agent and any future interface.

### Step 2.2: Evaluating Visualization Options

*   **Option 1: `MAV` (Model Activity Visualiser)**
    *   **Assessment:** `MAV` is a terminal-based tool for visualizing LLM activity *during generation*. While its plugin system for custom panels was promising, its dynamic nature and terminal limitations (especially for 2D/3D visualization and interactive editing) made it less ideal for the static tensor exploration and direct editing goals.
    *   **Decision:** Decided against `MAV` to pursue a more flexible web-based UI.

*   **Option 2: Custom Web UI (FastAPI + HTML/JS)**
    *   **Assessment:** A localhost web UI offers greater flexibility for custom visualizations, user interactions, and integrating with the existing Python backend.
    *   **Decision:** Proceed with building a web UI.

### Step 2.3: Initial Web UI Scaffolding

*   **Backend (`main_web.py`):**
    *   **Action:** Created a FastAPI application.
    *   **Functionality:**
        *   Initializes `AITensorExplorer` to load a model (currently hardcoded path, to be made configurable).
        *   Serves a static `index.html` page.
        *   Provides an API endpoint (`/api/tensors`) to list all available tensors from the loaded model.
    *   **Rationale:** FastAPI is a modern, fast Python web framework well-suited for this task.

*   **Frontend (`templates/index.html`, `static/styles.css`, `static/scripts.js`):**
    *   **Action:** Created basic HTML structure, CSS for styling, and JavaScript for initial interactivity.
    *   **Functionality:**
        *   Displays a list of tensors fetched from the `/api/tensors` endpoint.
        *   Includes a search filter for the tensor list.
        *   Placeholder sections for displaying detailed tensor information and editing controls.
    *   **Rationale:** Provides a foundational user interface.

## 3. Core Philosophy: Hypothesis-Driven Exploration and Editing

Before diving deeper into implementation, it's crucial to establish the guiding philosophy for this tool. The goal is not just to display tensors, but to create an **interactive model surgery toolkit**. This involves:

*   **Internalizing the End Product:** We envision a system where users can:
    1.  Formulate hypotheses about model behavior (e.g., "Why does the model say X?", "How can I make the model better at Y?", "What if I change parameter Z?").
    2.  Use visualizations and analytical tools to locate relevant tensors or patterns within tensors that might be responsible for or related to these hypotheses.
    3.  Visually inspect these tensors in an intuitive and aesthetically engaging way (the "Matrix-like" interface).
    4.  Perform targeted edits directly on the visual representation of these tensors (or through clearly linked controls).
    5.  Observe the impact of these edits by saving the modified model and re-testing it.
*   **Reverse Coding the Vision:** This end-product vision dictates our development. Each feature, from data fetching to visualization rendering to editing capabilities, is a step towards enabling this iterative loop of hypothesis, exploration, editing, and observation.
*   **Leveraging Existing Framework for the "HOW":**
    *   The existing `AITensorExplorer` and its associated tools (`TensorListTool`, `TensorStatisticsTool`, `TensorValuesTool`, `CodeAgent` for `investigate` commands) are fundamental to addressing the "HOW do we know what to edit?" question. These tools provide the analytical power to form hypotheses.
    *   The web UI will act as a sophisticated frontend to these capabilities, making them more accessible and pairing them with direct visual feedback and manipulation.
    *   `EnhancedTensorPatcher` remains the backend workhorse for applying the edits and saving new model versions.

## 4. Next Steps (Current Focus)

*   **Backend - API for Sliced Tensor Data:**
    *   Modify `/api/tensor/{tensor_name}` or create a new endpoint (e.g., `/api/tensor_slice/{tensor_name}`) in `main_web.py`.
    *   This endpoint must accept slice parameters (e.g., `?dim0_start=0&dim0_end=32&dim1_start=0&dim1_end=32` for a 2D slice).
    *   It will use `AITensorExplorer` (and its underlying `SafetensorsExplorer`) to load and return *all values* for the specifically requested tensor slice as JSON.
    *   This is crucial for detailed visualization, moving beyond small generic samples for the main display.
*   **Frontend - Plotly.js Heatmap for 2D Slices:**
    *   Integrate Plotly.js into `index.html`.
    *   In `static/scripts.js`:
        *   When a tensor is selected, provide UI elements (e.g., input fields) for the user to define which 2D slice to view (especially for 3D+ tensors). For 2D tensors, it might default to the full tensor if dimensions are manageable, or a default slice.
        *   Call the new backend API to fetch the specified tensor slice data.
        *   Use Plotly.js to render this 2D slice data as an interactive heatmap in the `#tensor-visualization` div.
        *   Implement hover-over-cell functionality to display the exact value and indices.
        *   Implement click-on-cell functionality to "select" a cell, preparing for future editing.
*   **Configuration:**
    *   The model path in `main_web.py` has been made configurable via the `MODEL_PATH` environment variable.

## 5. Future Goals (Post-Current Focus)

*   **Visualization:**
    *   Implement actual tensor visualizations in the frontend (e.g., heatmaps for 2D slices, distribution plots). Consider using a lightweight charting library.
*   **Editing Functionality:**
    *   Add UI elements (input fields, buttons) for specifying tensor modifications.
    *   Implement a backend API endpoint (e.g., `POST /api/tensor/{tensor_name}/edit`) in `main_web.py` that uses the existing `EnhancedTensorPatcher` logic to apply changes.
    *   Provide feedback in the UI about the success/failure of edit operations.
    *   Mechanism to refresh tensor data after edits.
*   **Binary Editing Representation:**
    *   Explore how to represent tensor data in a way that facilitates "binary editing" concepts (e.g., showing hex values, allowing direct modification of these if meaningful).
*   **Tensor Clustering Visualization:**
    *   Investigate methods to compute and display 2D/3D representations of tensor clusters (this may require dimensionality reduction techniques like PCA or t-SNE for visualization).

## 5.1. Refined Patching Workflow: Hex-Diff Patch Files

*   **Concept:** Instead of saving the entire modified model file after each edit or session of edits, the toolkit will generate a specialized "hex-diff patch file."
*   **Mechanism:**
    1.  As the user makes edits (e.g., changing specific values in a tensor slice, applying scaling operations), these modifications are tracked at a granular level, ideally down to the affected memory addresses/offsets and their new hex values.
    2.  When the user wants to save their work-in-progress or a set of changes, the system compiles these into a compact patch file. This file would essentially be a list of (tensor_name, offset, original_hex_sequence, new_hex_sequence) or similar, representing only the deltas.
    3.  This hex-diff patch file can be versioned, shared, and reapplied to the original base model at a later time to recreate the patched model state.
*   **Advantages:**
    *   **Efficiency:** Patch files will be significantly smaller than full model snapshots, especially for sparse edits.
    *   **Transparency & Auditability:** The patch file provides a clear record of exactly what was changed.
    *   **Modularity:** Patches could potentially be combined or selectively applied.
    *   **Reduced I/O:** Less disk writing during the editing session until a final "bake model" operation.
*   **Implementation Consideration:** This requires a robust backend mechanism to:
    *   Precisely map UI edits to tensor memory locations.
    *   Generate the hex diffs.
    *   Apply these diffs to a base model to materialize a fully patched model when needed (e.g., for final export or testing).
    *   The `EnhancedTensorPatcher` would need to be augmented or a new component created to handle this hex-level diffing and patching.

## 6. Technologies Used

*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** HTML, CSS, JavaScript
*   **Core Logic:** Existing Python classes (`AITensorExplorer`, `SafetensorsExplorer`, `EnhancedTensorPatcher`)

---
This document will be updated as the project evolves. 