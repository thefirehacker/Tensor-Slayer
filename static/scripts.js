document.addEventListener('DOMContentLoaded', () => {
    const tensorListUL = document.getElementById('tensor-list');
    const tensorDetailsDiv = document.getElementById('tensor-details-content');
    const tensorSearchInput = document.getElementById('tensor-search');
    let allTensors = []; // To store all fetched tensors for searching

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
        try {
            const response = await fetch(`/api/tensor/${encodeURIComponent(tensorName)}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const details = await response.json();
            renderTensorDetails(details);

        } catch (error) {
            tensorDetailsDiv.innerHTML = `<p>Error loading tensor details for ${tensorName}: ${error.message}</p>`;
            console.error("Fetch details error:", error);
        }
    }

    function renderTensorDetails(details) {
        const tensor = allTensors.find(t => t.name === details.name) || {}; // Get basic info like shape, dtype from allTensors if needed
        let html = `<h3>${details.name}</h3>`;
        html += '<table>';
        // Use basic info from the initial tensor list as fallback if not in detailed stats
        html += `<tr><th>Shape</th><td>${(details.stats && details.stats.shape) ? details.stats.shape.join(' x ') : tensor.shape.join(' x ')}</td></tr>`;
        html += `<tr><th>Data Type</th><td>${(details.stats && details.stats.dtype) ? details.stats.dtype : tensor.dtype}</td></tr>`;
        html += `<tr><th>Size (MB)</th><td>${tensor.size_mb ? tensor.size_mb.toFixed(2) : 'N/A'}</td></tr>`;
        
        if (details.stats) {
            html += `<tr><th>Min Value</th><td>${details.stats.min !== undefined ? details.stats.min.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Max Value</th><td>${details.stats.max !== undefined ? details.stats.max.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Mean</th><td>${details.stats.mean !== undefined ? details.stats.mean.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Std Dev</th><td>${details.stats.std !== undefined ? details.stats.std.toFixed(4) : 'N/A'}</td></tr>`;
            html += `<tr><th>Zeros %</th><td>${details.stats.zeros_percent !== undefined ? details.stats.zeros_percent.toFixed(2) + '%': 'N/A'}</td></tr>`;
        }
        html += '</table>';

        if (details.values_sample && details.values_sample.length > 0) {
            let sampleInfoStr = '';
            if (details.is_sampled) {
                sampleInfoStr = `(Sampled ${details.values_sample.length} elements`;
                if (details.sample_info && typeof details.sample_info === 'number') { // total_elements
                    sampleInfoStr += ` from ${details.sample_info} total`;
                }
                sampleInfoStr += ')';
            } else {
                sampleInfoStr = `(All ${details.values_sample.length} elements shown)`;
            }
            html += `<h4>Sample Values ${sampleInfoStr}:</h4>`;
            html += `<div class="value-view">${details.values_sample.map(v => typeof v === 'number' ? v.toFixed(6) : v).join('\n')}</div>`;
        } else {
            html += '<p>No sample values available or an error occurred fetching them.</p>';
        }
        
        // Placeholder for visualization (e.g., using a charting library)
        html += '<div id="tensor-visualization" style="margin-top: 20px; min-height: 200px; border: 1px dashed #ccc; padding:10px;">Tensor visualization placeholder</div>';
        
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