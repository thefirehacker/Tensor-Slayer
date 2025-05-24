import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import torch

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
async def get_tensor_details(tensor_name: str):
    if not ai_explorer_instance or not hasattr(ai_explorer_instance, 'explorer'):
        # This case should already be handled by the HTTPException if ai_explorer_instance is None
        print("Error: ai_explorer_instance is not properly initialized.")
        raise HTTPException(status_code=503, detail="Tensor explorer not initialized (ai_explorer_instance missing or invalid).")
    if tensor_name not in ai_explorer_instance.tensors:
        print(f"Error: Tensor '{tensor_name}' not found in ai_explorer_instance.tensors.")
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_name}' not found.")

    stats = None
    values_data = {}
    error_log = []

    try:
        print(f"Fetching stats for tensor: {tensor_name}")
        raw_stats = ai_explorer_instance.explorer.analyze_tensor(tensor_name)
        stats = {}
        if raw_stats:
            for key, value in raw_stats.items():
                if isinstance(value, torch.dtype):
                    stats[key] = str(value)
                elif isinstance(value, (list, tuple)) and any(isinstance(i, torch.dtype) for i in value):
                    # Handle cases where shape might be a list/tuple of dtypes (though less common for shape)
                    # More likely, shape is list of ints. Dtype is usually a single value.
                    stats[key] = [str(i) if isinstance(i, torch.dtype) else i for i in value]
                else:
                    stats[key] = value
        print(f"Successfully fetched and processed stats: {stats is not None}")
    except Exception as e:
        import traceback
        print(f"Error in analyze_tensor for {tensor_name}: {e}")
        error_log.append(f"Failed to analyze tensor: {str(e)}")
        # traceback.print_exc() # Keep this for server-side debugging if needed

    try:
        print(f"Fetching values for tensor: {tensor_name}")
        if hasattr(ai_explorer_instance, 'get_tensor_values'):
            values_data = ai_explorer_instance.get_tensor_values(tensor_name, max_elements=100)
            print(f"Successfully fetched values_data: {values_data is not None}")
            if "error" in values_data:
                print(f"Error reported by get_tensor_values for {tensor_name}: {values_data['error']}")
                error_log.append(f"Error from get_tensor_values: {values_data['error']}")
        else:
            print(f"Warning: ai_explorer_instance does not have get_tensor_values method.")
            error_log.append("get_tensor_values method not available on AITensorExplorer instance.")
            values_data = {"error": "Value fetching method not available."} # Ensure values_data is a dict
            
    except Exception as e:
        import traceback
        print(f"Error in get_tensor_values for {tensor_name}: {e}")
        error_log.append(f"Failed to get tensor values: {str(e)}")
        # traceback.print_exc() # Keep this for server-side debugging if needed
        if not isinstance(values_data, dict): # Ensure values_data is a dict even on exception
             values_data = {}
        values_data["error"] = f"Exception during value fetching: {str(e)}"


    # If both failed, or explorer was bad from start
    if stats is None and not values_data.get("values"):
        # If we have specific errors logged, include them
        detail_message = "Failed to retrieve any tensor details."
        if error_log:
            detail_message += " Errors: " + "; ".join(error_log)
        print(f"Critical failure for {tensor_name}: {detail_message}")
        # We are already inside a try block for the overall function,
        # so let the main exception handler below catch this if it's a total failure
        # Or, be more explicit:
        # raise HTTPException(status_code=500, detail=detail_message)
        # For now, let's construct a JSON response indicating failure for client
        return {
            "name": tensor_name,
            "stats": stats if stats is not None else {"error": "Statistics not available.", "shape": [], "dtype": "unknown"}, # Ensure stats is always a dict
            "values_sample": [],
            "is_sampled": False,
            "sample_info": None,
            "errors": error_log or ["Unknown error retrieving details"]
        }

    final_response = {
        "name": tensor_name,
        "stats": stats if stats is not None else {"error": "Statistics not available.", "shape": [], "dtype": "unknown"}, # Ensure stats is always a dict
        "values_sample": values_data.get("values", []),
        "is_sampled": values_data.get("is_sampled", False),
        "sample_info": values_data.get("sample_info") or values_data.get("total_elements"),
        "errors": error_log if error_log else None # Add error log to response
    }
    print(f"Preparing to send response for {tensor_name}: Stats loaded: {stats is not None}, Values loaded: {bool(values_data.get('values'))}")
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