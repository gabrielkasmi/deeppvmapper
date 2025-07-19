// Popup functions
function openPopup() {
    const popup = document.getElementById('infoPopup');
    if (popup) {
        popup.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }
}

function closePopup() {
    const popup = document.getElementById('infoPopup');
    if (popup) {
        popup.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Initialize popup functionality
document.addEventListener('DOMContentLoaded', function() {
    const popup = document.getElementById('infoPopup');
    
    // Show popup by default
    popup.style.display = 'block';
    
    // Close popup when clicking outside
    popup.addEventListener('click', function(e) {
        if (e.target === popup) {
            closePopup();
        }
    });
    
    // Close popup when pressing Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closePopup();
        }
    });
    
    // Ensure close button works
    const closeBtn = document.querySelector('.close-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', closePopup);
    }
});

// Fallback for immediate execution
if (document.readyState !== 'loading') {
    const popup = document.getElementById('infoPopup');
    if (popup) {
        popup.style.display = 'block';
    }
}

// DeepPVMapper Interactive Map
class DeepPVMapperMap {
    constructor() {
        console.log('DeepPVMapperMap constructor called');
        this.map = null;
        this.markerCluster = null;
        this.currentData = [];
        this.loadingElement = null;
        
        // Ensure pako is available
        if (typeof pako === 'undefined') {
            console.error('Pako library not loaded!');
            this.showError('Required library not loaded. Please refresh the page.');
            return;
        }
        
        this.init();
    }
    
    init() {
        console.log('Initializing map...');
        this.initMap();
        this.createLoadingIndicator();
        this.loadCompressedGeoJSONData();
    }
    
    initMap() {
        console.log('Initializing map container...');
        const mapContainer = document.getElementById('map');
        if (!mapContainer) {
            console.error('Map container not found!');
            return;
        }
        console.log('Map container found:', mapContainer);
        
        try {
            // Initialize the map centered on France
            this.map = L.map('map').setView([46.603354, 1.888334], 6);
            console.log('Map created:', this.map);
            
            // Add OpenStreetMap tiles
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }).addTo(this.map);
            console.log('Tile layer added');
            
            // Create a marker cluster group with standard settings
            this.markerCluster = L.markerClusterGroup({
                chunkedLoading: true,
                maxClusterRadius: 80,
                spiderfyOnMaxZoom: true,
                showCoverageOnHover: true,
                zoomToBoundsOnClick: true,
                disableClusteringAtZoom: 15, // Show individual markers at high zoom
                removeOutsideVisibleBounds: true
            });
            console.log('Marker cluster group created');
            
            this.map.addLayer(this.markerCluster);
            console.log('Marker cluster added to map');
        } catch (error) {
            console.error('Error initializing map:', error);
            this.showError('Failed to initialize map: ' + error.message);
        }
    }
    
    createLoadingIndicator() {
        this.loadingElement = document.createElement('div');
        this.loadingElement.className = 'loading-indicator';
        this.loadingElement.innerHTML = `
            <div class="loading-content">
                <div class="spinner"></div>
                <p>Loading PV installation data...</p>
                <p class="loading-details">Decompressing and processing data</p>
            </div>
        `;
        document.body.appendChild(this.loadingElement);
    }
    
    hideLoadingIndicator() {
        if (this.loadingElement) {
            this.loadingElement.style.opacity = '0';
            setTimeout(() => {
                if (this.loadingElement && this.loadingElement.parentNode) {
                    this.loadingElement.parentNode.removeChild(this.loadingElement);
                }
                this.loadingElement = null;
            }, 300);
        }
    }
    
    async loadCompressedGeoJSONData() {
        try {
            console.log('Loading compressed GeoJSON data...');
            console.log('Browser:', navigator.userAgent);
            console.log('Fetch available:', typeof fetch !== 'undefined');
            console.log('ArrayBuffer available:', typeof ArrayBuffer !== 'undefined');
            console.log('Pako available:', typeof pako !== 'undefined');
            
            console.log(`Browser: ${navigator.userAgent.split(' ')[0]}`);
            console.log(`Pako available: ${typeof pako !== 'undefined'}`);
            
            // Try compressed version first, then fallback to uncompressed
            let geojsonData = null;
            
            // First attempt: Try compressed version
            try {
                console.log('Trying compressed version...');
                geojsonData = await this.loadCompressedVersion();
                console.log('Compressed version loaded successfully');
            } catch (compressedError) {
                console.log('Compressed version failed, trying uncompressed:', compressedError.message);
                // Second attempt: Try uncompressed version
                try {
                    console.log('Trying uncompressed version...');
                    geojsonData = await this.loadUncompressedVersion();
                    console.log('Uncompressed version loaded successfully');
                } catch (uncompressedError) {
                    console.error('Both compressed and uncompressed versions failed');
                    throw new Error(`Compressed failed: ${compressedError.message}. Uncompressed failed: ${uncompressedError.message}`);
                }
            }
            
            if (!geojsonData || !geojsonData.features || !Array.isArray(geojsonData.features)) {
                console.error('Invalid GeoJSON structure:', geojsonData);
                throw new Error('Invalid GeoJSON structure');
            }
            
            console.log(`Parsed ${geojsonData.features.length} features`);
            console.log('Sample feature:', geojsonData.features[0]);
            
            this.currentData = geojsonData.features;
            this.displayData(this.currentData);
            this.updateStats();
            console.log('✓ Map ready!');
            
        } catch (error) {
            console.error('Error loading GeoJSON data:', error);
            console.error('Full error details:', {
                message: error.message,
                name: error.name,
                stack: error.stack
            });
            this.showError('Failed to load data. Please try refreshing the page. Error: ' + error.message);
        }
    }
    
    async loadCompressedVersion() {
        console.log('Attempting to load compressed version...');
        
        // Try different approaches for Firefox
        let response;
        try {
            // First attempt: Standard fetch with headers
            response = await fetch('../static/data/aggregated_data.geojson.gz', {
                method: 'GET',
                headers: {
                    'Accept': 'application/gzip, application/octet-stream, */*',
                    'Cache-Control': 'no-cache'
                }
            });
            console.log('Standard fetch successful:', response.status);
        } catch (fetchError) {
            console.log('Standard fetch failed, trying without headers:', fetchError);
            // Second attempt: Simple fetch without headers
            response = await fetch('../static/data/aggregated_data.geojson.gz');
            console.log('Simple fetch successful:', response.status);
        }
        
        console.log('Fetch response:', response);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Get the compressed data as ArrayBuffer
        console.log('Getting ArrayBuffer...');
        const compressedData = await response.arrayBuffer();
        console.log('Compressed data size:', compressedData.byteLength, 'bytes');
        console.log('ArrayBuffer type:', compressedData.constructor.name);
        
        // Decompress using pako with error handling
        console.log('Decompressing data with pako...');
        let decompressedData;
        try {
            // Convert ArrayBuffer to Uint8Array for pako
            const uint8Array = new Uint8Array(compressedData);
            console.log('Uint8Array created, length:', uint8Array.length);
            console.log('First few bytes:', uint8Array.slice(0, 10));
            
            decompressedData = pako.inflate(uint8Array, { to: 'string' });
            console.log('Pako decompression successful');
        } catch (pakoError) {
            console.error('Pako decompression error:', pakoError);
            console.error('Pako error details:', {
                message: pakoError.message,
                name: pakoError.name,
                stack: pakoError.stack
            });
            throw new Error('Failed to decompress data: ' + pakoError.message);
        }
        console.log('Decompressed data size:', decompressedData.length, 'characters');
        console.log('First 200 characters:', decompressedData.substring(0, 200));
        
        // Parse the JSON with error handling
        console.log('Parsing JSON...');
        let geojsonData;
        try {
            geojsonData = JSON.parse(decompressedData);
        } catch (jsonError) {
            console.error('JSON parsing error:', jsonError);
            console.error('JSON error details:', {
                message: jsonError.message,
                name: jsonError.name,
                stack: jsonError.stack
            });
            // Try to show where the JSON parsing failed
            const errorPosition = jsonError.message.match(/position (\d+)/);
            if (errorPosition) {
                const pos = parseInt(errorPosition[1]);
                console.error('JSON error at position:', pos);
                console.error('Context around error:', decompressedData.substring(Math.max(0, pos-50), pos+50));
            }
            throw new Error('Failed to parse JSON data: ' + jsonError.message);
        }
        
        return geojsonData;
    }
    
    async loadUncompressedVersion() {
        console.log('Attempting to load uncompressed version...');
        
        const response = await fetch('../static/data/aggregated_data.geojson');
        console.log('Uncompressed fetch response:', response);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const jsonText = await response.text();
        console.log('Uncompressed data size:', jsonText.length, 'characters');
        console.log('First 200 characters:', jsonText.substring(0, 200));
        
        try {
            const geojsonData = JSON.parse(jsonText);
            return geojsonData;
        } catch (jsonError) {
            console.error('JSON parsing error for uncompressed version:', jsonError);
            throw new Error('Failed to parse uncompressed JSON: ' + jsonError.message);
        }
    }
    
    displayData(data) {
        console.log('Displaying data, clearing existing markers...');
        // Clear existing markers
        this.markerCluster.clearLayers();
        
        console.log('Creating markers for', data.length, 'features');
        
        let validMarkers = 0;
        let errorCount = 0;
        
        data.forEach((feature, index) => {
            if (index % 5000 === 0) {
                console.log(`Processing feature ${index}/${data.length}`);
            }
            
            try {
                const properties = feature.properties || {};
                const coordinates = feature.geometry ? feature.geometry.coordinates : null;
                
                // Skip features without valid coordinates
                if (!coordinates || !Array.isArray(coordinates) || coordinates.length !== 2) {
                    if (errorCount < 5) { // Only log first few errors
                        console.log('Skipping invalid feature:', feature);
                        errorCount++;
                    }
                    return;
                }
                
                // Validate coordinates are numbers
                const lat = parseFloat(coordinates[1]);
                const lng = parseFloat(coordinates[0]);
                
                if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
                    if (errorCount < 5) {
                        console.log('Skipping feature with invalid coordinates:', coordinates);
                        errorCount++;
                    }
                    return;
                }
                
                // Create simple marker
                const marker = L.marker([lat, lng], {
                    icon: L.divIcon({
                        className: 'custom-marker',
                        html: '<div class="marker-dot"></div>',
                        iconSize: [8, 8],
                        iconAnchor: [4, 4]
                    })
                });
                
                // Create popup content
                const popupContent = this.createPopupContent(properties);
                marker.bindPopup(popupContent);
                
                // Add to cluster
                this.markerCluster.addLayer(marker);
                validMarkers++;
                
            } catch (markerError) {
                console.error('Error creating marker for feature', index, ':', markerError);
                errorCount++;
            }
        });
        
        console.log('Markers created and added to cluster. Valid markers:', validMarkers, 'Errors:', errorCount);
        this.hideLoadingIndicator();
    }
    
    createPopupContent(properties) {
        const name = properties.nom || 'Unknown location';
        const capacity = properties.installed_capacity || 0;
        const numSystems = properties.number_of_systems || 1;
        const year = properties.detection_year || 'Unknown';
        const error = properties.error;
        
        let errorText = '';
        if (error !== null && error !== undefined && !isNaN(error)) {
            errorText = `<p><strong>Detection Error:</strong> ${parseFloat(error).toFixed(1)}%</p>`;
        }
        
        return `
            <div class="pv-popup">
                <h4>${name}</h4>
                <p><strong>Total Capacity:</strong> ${parseFloat(capacity).toFixed(1)} kW</p>
                <p><strong>Number of Systems:</strong> ${parseInt(numSystems)}</p>
                <p><strong>Detection Year:</strong> ${year}</p>
                ${errorText}
            </div>
        `;
    }
    
    updateStats() {
        const totalLocations = this.currentData.length;
        const totalCapacity = this.currentData.reduce((sum, feature) => {
            return sum + (parseFloat(feature.properties?.installed_capacity) || 0);
        }, 0);
        const avgCapacity = totalLocations > 0 ? totalCapacity / totalLocations : 0;
        
        console.log('Statistics:', {
            totalLocations,
            totalCapacity: totalCapacity.toFixed(1) + ' kW',
            avgCapacity: avgCapacity.toFixed(1) + ' kW'
        });
    }
    
    showError(message) {
        this.hideLoadingIndicator();
        
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.innerHTML = `
            <div class="error-content">
                <h3>Error</h3>
                <p>${message}</p>
                <button onclick="location.reload()">Retry</button>
            </div>
        `;
        document.body.appendChild(errorElement);
    }
}

// Initialize the map when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing map...');
    try {
        window.deepPVMapperMap = new DeepPVMapperMap();
    } catch (error) {
        console.error('Error initializing DeepPVMapperMap:', error);
        // Show error message
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.innerHTML = `
            <div class="error-content">
                <h3>Initialization Error</h3>
                <p>Failed to initialize the map: ${error.message}</p>
                <button onclick="location.reload()">Retry</button>
            </div>
        `;
        document.body.appendChild(errorElement);
    }
}); 