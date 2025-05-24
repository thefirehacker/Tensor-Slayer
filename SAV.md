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

## 3. Next Steps (Current Focus)

*   **Backend:**
    *   Implement a new API endpoint in `main_web.py`: `GET /api/tensor/{tensor_name}`.
        *   This endpoint will return detailed statistics (from `analyze_tensor`) and a sample of values (e.g., using `get_tensor_values` or a relevant tool from `AITensorExplorer`) for the specified tensor.
*   **Frontend:**
    *   Update `static/scripts.js` to call the new `/api/tensor/{tensor_name}` endpoint when a tensor is selected.
    *   Display the fetched statistics and values in the "Tensor Details" panel of `index.html`.
*   **Configuration:**
    *   Make the model path in `main_web.py` more configurable (e.g., environment variable or a simple configuration file/UI element).

## 4. Future Goals (Post-Current Focus)

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

## 5. Technologies Used

*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** HTML, CSS, JavaScript
*   **Core Logic:** Existing Python classes (`AITensorExplorer`, `SafetensorsExplorer`, `EnhancedTensorPatcher`)

---
This document will be updated as the project evolves. 