import os
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import torch
import numpy as np
from typing import Optional

# Assuming your existing explorers are in the same directory or accessible in PYTHONPATH
from ai_tensor_explorer import AITensorExplorer
from safetensors_explorer_cli import SafetensorsExplorer

# Configuration
# Read model path from environment variable or use a default
DEFAULT_MODEL_PATH_FALLBACK = "./Qwen_0.6B" # Fallback if MODEL_PATH env var is not set
MODEL_DIRECTORY = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH_FALLBACK)

print(f"Attempting to load model from: {MODEL_DIRECTORY}")

# Attempt to find a .safetensors file in the MODEL_DIRECTORY
model_files = list(Path(MODEL_DIRECTORY).glob('*.safetensors'))
if not model_files:
    print(f"Warning: No .safetensors files found in {MODEL_DIRECTORY}. The application might not work correctly.")
    initial_weight_file_path = ""
else:
    initial_weight_file_path = str(model_files[0])


app = FastAPI()

# Mount static files and templates
# Ensure these directories exist: ./static and ./templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)
if not templates_dir.exists():
    templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global variable to hold the explorer instance
# This is a simple way for a single-user, local app.
# For multi-user or production, you'd need a more robust way to manage state.
try:
    # AITensorExplorer expects a list of weight files.
    # We'll use the first .safetensors file found.
    ai_explorer_instance = AITensorExplorer(weight_files=[initial_weight_file_path] if initial_weight_file_path else [])
except Exception as e:
    print(f"Error initializing AITensorExplorer: {e}")
    print("Ensure your model path is correct and .safetensors files are present.")
    # Assign a None or dummy object if initialization fails, to allow app to at least start
    # and potentially provide an error message in the UI.
    ai_explorer_instance = None 

MAX_DIM_SIZE_FOR_FULL_2D_RETURN = 128 # Max dimension size for auto-returning full 2D tensor
DEFAULT_SLICE_SIZE = 32 # Default slice size if tensor is too large and no slice params given

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    model_display_path = "N/A"
    explorer_ready = False
    if ai_explorer_instance and hasattr(ai_explorer_instance, 'weight_files') and ai_explorer_instance.weight_files:
        # ai_explorer_instance.model_path is the directory, weight_files[0] is the specific file.
        # Let's use the directory as it's more general if multiple files were ever supported by AITensorExplorer's constructor in a merged way.
        # Or, just show the first file used for initialization.
        model_display_path = ai_explorer_instance.model_path 
        if not Path(model_display_path).is_dir(): # If model_path is actually the file path from older AITensorExplorer version
             model_display_path = str(Path(ai_explorer_instance.weight_files[0]).parent)
        explorer_ready = True if ai_explorer_instance.tensors else False # Check if tensors got loaded
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "explorer_ready": explorer_ready, "model_display_path": model_display_path}
    )

@app.get("/api/tensors")
async def list_tensors():
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'tensors'):
        raise HTTPException(status_code=503, detail="Tensor explorer not initialized or no tensors loaded.")
    
    tensor_list = []
    for name, info in ai_explorer_instance.tensors.items():
        tensor_list.append({
            "name": name,
            "shape": info.get('shape', 'N/A'),
            "dtype": info.get('dtype', 'N/A'),
            "size_mb": info.get('size_mb', 0)
        })
    return {"tensors": tensor_list}

@app.get("/api/tensor/{tensor_name}")
async def get_tensor_details(
    tensor_name: str,
    # Optional query parameters for slicing
    # For simplicity, supporting up to 2 dimensions for slicing via query params initially
    # These define a rectangular slice for the first two dimensions if they exist.
    dim0_start: Optional[int] = Query(None, alias="d0s"),
    dim0_end: Optional[int] = Query(None, alias="d0e"),
    dim1_start: Optional[int] = Query(None, alias="d1s"),
    dim1_end: Optional[int] = Query(None, alias="d1e"),
    # For higher dimensions, we'd need a more complex slicing mechanism or expect user to slice N-2 dims
    # For now, if >2D, these params apply to the first 2 dims, and other dims are fully included for the slice.
    # A full implementation might take a string like "0:32,0:32,:,5" for more general slicing.
):
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'explorer'):
        print("Error: ai_explorer_instance is not properly initialized.")
        raise HTTPException(status_code=503, detail="Tensor explorer not initialized.")
    if tensor_name not in ai_explorer_instance.tensors:
        print(f"Error: Tensor '{tensor_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_name}' not found.")

    stats = None
    processed_stats = {}
    tensor_slice_data = []
    slice_applied_info = "No slicing applied or requested for visualization."
    error_log = []

    try:
        print(f"Fetching stats for tensor: {tensor_name}")
        raw_stats = ai_explorer_instance.explorer.analyze_tensor(tensor_name)
        if raw_stats:
            for key, value in raw_stats.items():
                if isinstance(value, torch.dtype):
                    processed_stats[key] = str(value)
                else:
                    processed_stats[key] = value
        stats = processed_stats # Use the processed stats
        print(f"Successfully fetched and processed stats: {stats is not None}")
    except Exception as e:
        print(f"Error in analyze_tensor for {tensor_name}: {e}")
        error_log.append(f"Failed to analyze tensor: {str(e)}")

    try:
        print(f"Loading full tensor for potential slicing: {tensor_name}")
        # Load the full tensor first using the explorer's method
        # SafetensorsExplorer.load_tensor() should return a torch tensor
        full_tensor = ai_explorer_instance.explorer.load_tensor(tensor_name)
        original_shape = list(full_tensor.shape)

        slicing_params_provided = dim0_start is not None and dim0_end is not None
        
        current_slice_obj = [slice(None)] * full_tensor.dim() # Start with full slices for all dims

        if slicing_params_provided:
            # Apply dim0 slicing if params are valid
            d0s = max(0, dim0_start if dim0_start is not None else 0)
            d0e = min(original_shape[0], dim0_end if dim0_end is not None else original_shape[0])
            if d0s < d0e:
                current_slice_obj[0] = slice(d0s, d0e)
            
            # Apply dim1 slicing if tensor has at least 2 dims and params are valid
            if full_tensor.dim() > 1 and dim1_start is not None and dim1_end is not None:
                d1s = max(0, dim1_start if dim1_start is not None else 0)
                d1e = min(original_shape[1], dim1_end if dim1_end is not None else original_shape[1])
                if d1s < d1e:
                    current_slice_obj[1] = slice(d1s, d1e)
            slice_applied_info = f"Applied slice: {tuple(current_slice_obj)}"
        elif full_tensor.dim() == 2 and (original_shape[0] > MAX_DIM_SIZE_FOR_FULL_2D_RETURN or original_shape[1] > MAX_DIM_SIZE_FOR_FULL_2D_RETURN):
            # Default slicing for large 2D tensors if no params given
            current_slice_obj[0] = slice(0, min(original_shape[0], DEFAULT_SLICE_SIZE))
            current_slice_obj[1] = slice(0, min(original_shape[1], DEFAULT_SLICE_SIZE))
            slice_applied_info = f"Default slice for large 2D: {tuple(current_slice_obj)}"
        elif full_tensor.dim() > 2: # For >2D tensors, if no slice given, take a default slice from first 2 dims
            current_slice_obj[0] = slice(0, min(original_shape[0], DEFAULT_SLICE_SIZE if original_shape[0] > DEFAULT_SLICE_SIZE else original_shape[0]))
            if full_tensor.dim() > 1:
                current_slice_obj[1] = slice(0, min(original_shape[1], DEFAULT_SLICE_SIZE if original_shape[1] > DEFAULT_SLICE_SIZE else original_shape[1]))
            # For remaining dimensions, we take only the 0-th index to make it 2D for heatmap
            for i in range(2, full_tensor.dim()):
                current_slice_obj[i] = 0 # Take the first element along higher dimensions
            slice_applied_info = f"Default 2D slice from higher-dim tensor: {tuple(current_slice_obj)}"
        
        # Perform the actual slicing
        sliced_tensor_view = full_tensor[tuple(current_slice_obj)]
        
        # Ensure the result for visualization is 2D or can be squeezed to 2D if higher dims were all size 1 after slicing
        if sliced_tensor_view.dim() > 2:
            try:
                # Attempt to squeeze out dimensions of size 1 to get to 2D if possible
                # This handles cases where higher dims were sliced to index 0
                squeezed_view = sliced_tensor_view.squeeze()
                if squeezed_view.dim() == 2:
                    sliced_tensor_view = squeezed_view
                elif squeezed_view.dim() == 1 and sliced_tensor_view.shape[0] == 1: # e.g. shape [1, N, 1, 1] -> [N]
                     # if original slice was like [0, 0:N, 0, 0], squeeze might make it 1D.
                     # Plotly heatmap typically wants 2D. We might need to reshape or handle 1D differently.
                     # For now, if it becomes 1D, we might represent it as a single row/column heatmap.
                     # If squeezed_view is 1D, convert to a 2D list with one row
                     sliced_tensor_view = squeezed_view.unsqueeze(0) # Make it [1, N]
                elif squeezed_view.dim() == 0: # Scalar result from slicing
                    sliced_tensor_view = squeezed_view.unsqueeze(0).unsqueeze(0) # Make it [[value]]
                # If still > 2D after squeeze, the slice was not effectively 2D, log and error or send as is if client can handle
                # For now, we aim for 2D for heatmap visualization
                if sliced_tensor_view.dim() > 2:
                     error_log.append(f"Slice result for visualization is still >2D: {list(sliced_tensor_view.shape)}. Heatmap may not display as expected.")
            except Exception as e_squeeze:
                error_log.append(f"Error trying to squeeze tensor slice: {str(e_squeeze)}")

        # Convert to Python list of lists (for JSON)
        # Ensure tensor is on CPU before converting to numpy then list
        
        # Handle BFloat16 conversion before numpy()
        if sliced_tensor_view.dtype == torch.bfloat16:
            print(f"Converting tensor slice from bfloat16 to float32 for serialization.")
            sliced_tensor_view = sliced_tensor_view.to(torch.float32)
            
        tensor_slice_data = sliced_tensor_view.cpu().numpy().tolist()
        print(f"Successfully sliced tensor. Slice shape: {list(sliced_tensor_view.shape)}")

    except Exception as e:
        print(f"Error in tensor loading/slicing for {tensor_name}: {e}")
        import traceback
        traceback.print_exc()
        error_log.append(f"Failed to load/slice tensor: {str(e)}")

    final_response = {
        "name": tensor_name,
        "stats": stats if stats is not None else {"error": "Statistics not available.", "shape": original_shape if 'original_shape' in locals() else [], "dtype": "unknown"},
        "tensor_slice_data": tensor_slice_data, # This is the primary data for visualization
        "slice_applied_info": slice_applied_info,
        "original_shape": original_shape if 'original_shape' in locals() else [],
        "slice_shape": list(sliced_tensor_view.shape) if 'sliced_tensor_view' in locals() and hasattr(sliced_tensor_view, 'shape') else [],
        "errors": error_log if error_log else None
    }
    print(f"Preparing to send response for {tensor_name}: Stats loaded: {stats is not None}, Slice data generated: {bool(tensor_slice_data)}")
    return final_response

# TODO: Add more endpoints:
# - GET /api/tensor/{tensor_name} -> get detailed info, stats, and values for a tensor
# - POST /api/tensor/{tensor_name}/edit -> apply modifications to a tensor

if __name__ == "__main__":
    # Make sure to have `uvicorn` and `fastapi` installed:
    # pip install uvicorn fastapi python-multipart jinja2
    # Run with: python main_web.py
    # Or for development with auto-reload: uvicorn main_web:app --reload
    
    # Check if initial_weight_file_path is valid before trying to run
    if not initial_weight_file_path and (not ai_explorer_instance or not ai_explorer_instance.weight_files):
        print("CRITICAL: No model file was loaded. The AITensorExplorer could not be initialized.")
        print(f"Please ensure a .safetensors file exists in {MODEL_DIRECTORY} or update MODEL_DIRECTORY.")
        print("Application will start but tensor operations will fail.")
        # Optionally, exit here if a model is strictly required:
        # import sys
        # sys.exit("Exiting due to missing model file.")

    uvicorn.run(app, host="127.0.0.1", port=8000) 