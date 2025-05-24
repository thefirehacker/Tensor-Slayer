document.addEventListener('DOMContentLoaded', () => {
    const tensorListUL = document.getElementById('tensor-list');
    const tensorDetailsDiv = document.getElementById('tensor-details-content');
    const tensorSearchInput = document.getElementById('tensor-search');
    
    // Slice control inputs
    const d0sInput = document.getElementById('d0s_input');
    const d0eInput = document.getElementById('d0e_input');
    const d1sInput = document.getElementById('d1s_input');
    const d1eInput = document.getElementById('d1e_input');
    const applySliceBtn = document.getElementById('apply-slice-btn');
    let currentSelectedTensorName = null; // To store the currently selected tensor name

    let allTensors = []; // To store all fetched tensors for searching
    let currentTensorSliceParams = { d0s: 0, d0e: 32, d1s: 0, d1e: 32 }; // Default slice

    function updateSliceParams() {
        currentTensorSliceParams.d0s = parseInt(d0sInput.value) || 0;
        currentTensorSliceParams.d0e = parseInt(d0eInput.value) || 32;
        currentTensorSliceParams.d1s = parseInt(d1sInput.value) || 0;
        currentTensorSliceParams.d1e = parseInt(d1eInput.value) || 32;

        // Basic validation to ensure start < end
        if (currentTensorSliceParams.d0e <= currentTensorSliceParams.d0s) {
            currentTensorSliceParams.d0e = currentTensorSliceParams.d0s + 1;
            d0eInput.value = currentTensorSliceParams.d0e; // Update UI
        }
        if (currentTensorSliceParams.d1e <= currentTensorSliceParams.d1s) {
            currentTensorSliceParams.d1e = currentTensorSliceParams.d1s + 1;
            d1eInput.value = currentTensorSliceParams.d1e; // Update UI
        }
    }

    if (applySliceBtn) {
        applySliceBtn.addEventListener('click', () => {
            if (currentSelectedTensorName) {
                updateSliceParams(); // Get latest values from inputs
                fetchTensorDetails(currentSelectedTensorName); // Re-fetch with new slice params
            }
        });
    }
    
    // Update input fields if currentTensorSliceParams changes (e.g. due to validation)
    function syncSliceInputsToParams(){
        if(d0sInput) d0sInput.value = currentTensorSliceParams.d0s;
        if(d0eInput) d0eInput.value = currentTensorSliceParams.d0e;
        if(d1sInput) d1sInput.value = currentTensorSliceParams.d1s;
        if(d1eInput) d1eInput.value = currentTensorSliceParams.d1e;
    }

    async function fetchTensors() {
        try {
            const response = await fetch('/api/tensors');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            allTensors = data.tensors;
            renderTensorList(allTensors);
        } catch (error) {
            tensorListUL.innerHTML = `<li>Error loading tensors: ${error.message}</li>`;
            console.error("Fetch error:", error);
        }
    }

    function renderTensorList(tensorsToRender) {
        tensorListUL.innerHTML = ''; // Clear existing list
        if (tensorsToRender.length === 0) {
            tensorListUL.innerHTML = '<li>No tensors found.</li>';
            return;
        }
        tensorsToRender.forEach(tensor => {
            const li = document.createElement('li');
            li.textContent = `${tensor.name} (Shape: ${tensor.shape.join('x')}, ${tensor.size_mb.toFixed(2)} MB)`;
            li.dataset.tensorName = tensor.name;
            li.addEventListener('click', () => {
                currentSelectedTensorName = tensor.name; // Store selected tensor name
                // When a new tensor is selected, reset slice params to default or last valid for that tensor?
                // For now, just use current slice params, which might be from a previous tensor.
                // Or, perhaps reset to default slice and update inputs:
                currentTensorSliceParams = { d0s: 0, d0e: 32, d1s: 0, d1e: 32 };
                syncSliceInputsToParams();
                fetchTensorDetails(tensor.name);
                // Highlight selected tensor
                document.querySelectorAll('#tensor-list li').forEach(item => item.classList.remove('selected'));
                li.classList.add('selected');
            });
            tensorListUL.appendChild(li);
        });
    }

    tensorSearchInput.addEventListener('input', (event) => {
        const searchTerm = event.target.value.toLowerCase();
        const filteredTensors = allTensors.filter(tensor => 
            tensor.name.toLowerCase().includes(searchTerm)
        );
        renderTensorList(filteredTensors);
    });

    async function fetchTensorDetails(tensorName) {
        tensorDetailsDiv.innerHTML = `<p>Loading details for ${tensorName}...</p>`;
        updateSliceParams(); // Ensure params are up-to-date before fetching
        
        const queryParams = new URLSearchParams({
            d0s: currentTensorSliceParams.d0s,
            d0e: currentTensorSliceParams.d0e,
            d1s: currentTensorSliceParams.d1s,
            d1e: currentTensorSliceParams.d1e
        }).toString();

        try {
            const response = await fetch(`/api/tensor/${encodeURIComponent(tensorName)}?${queryParams}`);
            if (!response.ok) {
                const errorData = await response.json();
                tensorDetailsDiv.innerHTML = `<p>Error loading tensor details for ${tensorName}: ${errorData.detail || 'Unknown error'}</p>`;
                if(errorData.errors) {
                    tensorDetailsDiv.innerHTML += `<p>Server errors: ${errorData.errors.join('; ')}</p>`;
                }
                console.error("Fetch details error data:", errorData);
                return; // Stop further processing if error
            }
            const details = await response.json();
            renderTensorDetails(details);
            if (details.tensor_slice_data && details.tensor_slice_data.length > 0) {
                renderHeatmap(details.tensor_slice_data, 'tensor-visualization', details.name, details.slice_applied_info);
            } else {
                document.getElementById('tensor-visualization').innerHTML = '<p>No slice data available for heatmap visualization.</p>';
            }

        } catch (error) {
            tensorDetailsDiv.innerHTML = `<p>Error loading tensor details for ${tensorName}: ${error.message}</p>`;
            console.error("Fetch details error:", error);
        }
    }

    function renderTensorDetails(details) {
        const tensorBasicInfo = allTensors.find(t => t.name === details.name) || {}; 
        let html = `<h3>${details.name}</h3>`;

        if (details.errors && details.errors.length > 0) {
            html += `<p style="color: red;"><strong>Errors:</strong> ${details.errors.join('; ')}</p>`;
        }

        html += '<table>';
        const originalShape = details.original_shape && details.original_shape.length > 0 ? details.original_shape : tensorBasicInfo.shape;
        html += `<tr><th>Original Shape</th><td>${originalShape ? originalShape.join(' x ') : 'N/A'}</td></tr>`;
        html += `<tr><th>Data Type</th><td>${(details.stats && details.stats.dtype) ? details.stats.dtype : tensorBasicInfo.dtype || 'N/A'}</td></tr>`;
        html += `<tr><th>Size (MB)</th><td>${tensorBasicInfo.size_mb ? tensorBasicInfo.size_mb.toFixed(2) : 'N/A'}</td></tr>`;
        
        if (details.stats && !details.stats.error) {
            html += `<tr><th>Min Value</th><td>${details.stats.min !== undefined ? details.stats.min.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Max Value</th><td>${details.stats.max !== undefined ? details.stats.max.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Mean</th><td>${details.stats.mean !== undefined ? details.stats.mean.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Std Dev</th><td>${details.stats.std !== undefined ? details.stats.std.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Zeros %</th><td>${details.stats.zeros_percent !== undefined ? details.stats.zeros_percent.toFixed(2) + '%': 'N/A'}</td></tr>`;
        } else if (details.stats && details.stats.error) {
            html += `<tr><th>Statistics</th><td>Error loading stats: ${details.stats.error}</td></tr>`;
        }
        html += '</table>';

        html += `<h4>Slice Details:</h4>`;
        html += `<p>Slice Applied: ${details.slice_applied_info || 'N/A'}</p>`;
        html += `<p>Slice Shape: ${details.slice_shape && details.slice_shape.length > 0 ? details.slice_shape.join(' x ') : 'N/A'}</p>`;
        
        // The raw value view might be less useful now that we have heatmap, but can keep for direct small slice view
        if (details.tensor_slice_data && details.tensor_slice_data.length > 0 && details.slice_shape && details.slice_shape.reduce((a,b) => a*b, 1) < 500) { // Only show raw for small slices
            html += '<h4>Raw Slice Values (if small):</h4>';
            // Simple formatting for 2D or 1D data
            let formatted_values = "";
            if (Array.isArray(details.tensor_slice_data[0])) { // 2D data
                formatted_values = details.tensor_slice_data.map(row => 
                    '[' + row.map(val => (typeof val === 'number' ? val.toFixed(4) : val)).join(', ') + ']'
                ).join('\n');
            } else { // 1D data (after potential squeeze)
                formatted_values = '[' + details.tensor_slice_data.map(val => (typeof val === 'number' ? val.toFixed(4) : val)).join(', ') + ']';
            }
            html += `<div class="value-view">${formatted_values}</div>`;
        }
        
        // Ensure this div is created empty for the heatmap or its alternative message
        html += '<div id="tensor-visualization" style="margin-top: 20px; min-height: 300px; border: 1px dashed #ccc; padding:10px;"></div>';
        
        // Placeholder for editing controls
        html += '<div id="tensor-editing" style="margin-top: 20px;">';
        html += '<h4>Edit Tensor</h4>';
        html += '<p>Editing controls will go here (e.g., operation, value, apply button).</p>';
        html += '</div>';

        tensorDetailsDiv.innerHTML = html;
    }

    // Initial fetch
    fetchTensors();
});

// Placeholder for Plotly heatmap rendering function
function renderHeatmap(data, elementId, tensorName, sliceInfo) {
    const vizDiv = document.getElementById(elementId);
    if (!vizDiv) return;

    // Check if data is 2D array, plotly expects z to be array of arrays for heatmap
    let z_data = data;
    if (!Array.isArray(data) || (data.length > 0 && !Array.isArray(data[0]))) {
        // If data is 1D, wrap it into a 2D array (single row heatmap)
        z_data = [data]; 
    }
    
    // Ensure all inner arrays have same length for a proper heatmap (for Plotly)
    // This might be needed if backend sometimes returns jagged arrays for 1D -> 2D conversion
    if (z_data.length > 0 && Array.isArray(z_data[0])){
        const firstRowLength = z_data[0].length;
        for(let i = 1; i < z_data.length; i++) {
            if (z_data[i].length !== firstRowLength) {
                // This indicates an issue, either with data or how it was prepared.
                // For now, we'll truncate/pad to make it rectangular, but ideally backend ensures consistent shape.
                // Or, Plotly might handle it gracefully / error out. Better to ensure rectilinearity.
                // console.warn("Heatmap data rows have inconsistent lengths. Adjusting for Plotly.");
                // This is a simplistic fix, might hide underlying issues.
                if (z_data[i].length > firstRowLength) z_data[i] = z_data[i].slice(0, firstRowLength);
                else while(z_data[i].length < firstRowLength) z_data[i].push(null); // Pad with null
            }
        }
    }

    const trace = {
        z: z_data,
        type: 'heatmap',
        colorscale: 'Viridis', // 'Plasma', 'Jet', 'Hot', 'Greens', etc.
        reversescale: false,
        hoverongaps: false,
        colorbar: {title: 'Value', titleside: 'right'}
    };

    const layout = {
        title: `Heatmap: ${tensorName} (${sliceInfo || 'Full View'})`,
        xaxis: { title: 'Dim 1 Index' },
        yaxis: { title: 'Dim 0 Index', autorange: 'reversed' }, // Often you want 0,0 at top-left
        width: vizDiv.clientWidth > 50 ? vizDiv.clientWidth - 20 : 500, // Responsive width
        height: (z_data.length * 20 > 300 ? z_data.length * 20 : 300) + 100, // Dynamic height based on rows, with min
        margin: { l: 50, r: 50, b: 100, t: 100, pad: 4 }
    };
    
    // Check if Plotly is available
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(elementId, [trace], layout, {responsive: true});
    } else {
        vizDiv.innerHTML = '<p>Plotly.js not loaded. Heatmap cannot be displayed.</p>';
    }
} 