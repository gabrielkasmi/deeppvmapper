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

function closeCityPopup() {
    if (window.deepPVMapperMap) {
        window.deepPVMapperMap.hideCityPopup();
    }
}

// Search functionality
function searchCity() {
    const searchInput = document.getElementById('citySearch');
    const query = searchInput.value.trim().toLowerCase();
    
    if (query.length === 0) {
        clearSearch();
        return;
    }
    
    if (window.deepPVMapperMap) {
        window.deepPVMapperMap.searchCities(query);
    }
}

function clearSearch() {
    const searchInput = document.getElementById('citySearch');
    const clearBtn = document.querySelector('.clear-search-btn');
    const resultsContainer = document.getElementById('searchResults');
    
    searchInput.value = '';
    clearBtn.style.display = 'none';
    resultsContainer.style.display = 'none';
    
    if (window.deepPVMapperMap) {
        window.deepPVMapperMap.clearSearch();
    }
}

// Add event listeners for search
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('citySearch');
    const clearBtn = document.querySelector('.clear-search-btn');
    
    // Search on Enter key
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchCity();
        }
    });
    
    // Show/hide clear button based on input
    searchInput.addEventListener('input', function() {
        clearBtn.style.display = this.value.length > 0 ? 'flex' : 'none';
        
        // Auto-search after 2 characters
        if (this.value.length >= 2) {
            searchCity();
        } else if (this.value.length === 0) {
            clearSearch();
        }
    });
    
    // Clear search on Escape key
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            clearSearch();
            this.blur();
        }
    });
});

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
        this.currentCityPopup = null;
        this.allMarkers = []; // Store all markers for search functionality
        this.searchResults = []; // Store current search results
        
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
        this.allMarkers = []; // Clear stored markers
        
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
                
                // Add click event for custom popup
                marker.on('click', () => {
                    this.showCityPopup(properties, [lat, lng]);
                });
                
                // Store marker with properties for search
                this.allMarkers.push({
                    marker: marker,
                    properties: properties,
                    coordinates: [lat, lng]
                });
                
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
        const capacity = properties.installed_capacity;
        const numSystems = properties.number_of_systems;
        const year = properties.detection_year || 'Unknown';
        const error = properties.error;
        const precision = properties.precision;
        const recall = properties.recall;
        const f1 = properties.f1;
        const department = properties.department;
        
        // Check if there are no detections
        const hasDetections = capacity !== null && capacity !== undefined && capacity !== 'None' && 
                             numSystems !== null && numSystems !== undefined && numSystems !== 'None';
        
        // Get error color class
        let errorColorClass = '';
        if (error !== null && error !== undefined && !isNaN(error)) {
            const errorValue = parseFloat(error);
            if (errorValue < 15) {
                errorColorClass = 'error-green';
            } else if (errorValue < 25) {
                errorColorClass = 'error-light-green';
            } else if (errorValue < 50) {
                errorColorClass = 'error-orange';
            } else {
                errorColorClass = 'error-red';
            }
        }
        
        // Check if we should show plots (more than 30 systems and has detections)
        const showPlots = hasDetections && numSystems > 30;
        
        return `
            <div class="city-popup">
                <div class="city-popup-header">
                    <h3>${name} (${year})</h3>
                    <button class="city-close-btn" onclick="closeCityPopup()">×</button>
                </div>
                <div class="city-popup-body">
                    <div class="city-section">
                        <h4>PV Installation Statistics</h4>
                        ${hasDetections ? `
                        <div class="city-stats">
                            <div class="stat-item">
                                <span class="stat-label">Total Installed Capacity</span>
                                <span class="stat-value">${parseFloat(capacity).toFixed(1)} kWp</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Number of Systems</span>
                                <span class="stat-value">${parseInt(numSystems)}</span>
                            </div>
                        </div>
                        ` : `
                        <div class="no-detections">
                            <p>No detections</p>
                        </div>
                        `}
                    </div>
                    
                    ${hasDetections ? `
                    <div class="city-section">
                        <h4>Consensus Metrics</h4>
                        <div class="city-stats">
                            ${error !== null && error !== undefined && !isNaN(error) ? `
                            <div class="stat-item">
                                <span class="stat-label">APE (Detection Error)</span>
                                <span class="stat-value ${errorColorClass}">${parseFloat(error).toFixed(1)}%</span>
                            </div>
                            ` : ''}
                            ${precision !== null && precision !== undefined && !isNaN(precision) ? `
                            <div class="stat-item">
                                <span class="stat-label">Precision</span>
                                <span class="stat-value">${parseFloat(precision).toFixed(2)}</span>
                            </div>
                            ` : ''}
                            ${recall !== null && recall !== undefined && !isNaN(recall) ? `
                            <div class="stat-item">
                                <span class="stat-label">Recall</span>
                                <span class="stat-value">${parseFloat(recall).toFixed(2)}</span>
                            </div>
                            ` : ''}
                            ${f1 !== null && f1 !== undefined && !isNaN(f1) ? `
                            <div class="stat-item">
                                <span class="stat-label">F1 Score</span>
                                <span class="stat-value">${parseFloat(f1).toFixed(2)}</span>
                            </div>
                            ` : ''}
                            ${department ? `
                            <div class="stat-item">
                                <span class="stat-label">Department</span>
                                <span class="stat-value">${department}</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                    ` : ''}
                    
                    ${showPlots ? `
                    <div class="city-section">
                        <h4>PV Systems Characteristics</h4>
                        <div class="plots-container">
                            <div class="plot-item">
                                <h5>Capacity Distribution</h5>
                                <canvas id="capacity-chart-${name.replace(/\s+/g, '-')}" width="300" height="120"></canvas>
                            </div>
                            <div class="plot-item">
                                <h5>Tilt Angles</h5>
                                <canvas id="tilt-chart-${name.replace(/\s+/g, '-')}" width="300" height="120"></canvas>
                            </div>
                            <div class="plot-item">
                                <h5>Azimuth (°)</h5>
                                <canvas id="azimuth-chart-${name.replace(/\s+/g, '-')}" width="300" height="120"></canvas>
                            </div>
                        </div>
                    </div>
                    ` : ''}
                </div>
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
    
    showCityPopup(properties, coordinates) {
        // Remove existing city popup if any
        this.hideCityPopup();
        
        // Create popup content
        const popupContent = this.createPopupContent(properties);
        
        // Create popup element
        const popupElement = document.createElement('div');
        popupElement.className = 'city-popup-overlay';
        popupElement.innerHTML = popupContent;
        
        // Add click outside to close
        popupElement.addEventListener('click', (e) => {
            if (e.target === popupElement) {
                this.hideCityPopup();
            }
        });
        
        // Add to body
        document.body.appendChild(popupElement);
        
        // Store reference
        this.currentCityPopup = popupElement;
        
        // Show popup with animation
        setTimeout(() => {
            popupElement.style.opacity = '1';
            const popupContent = popupElement.querySelector('.city-popup');
            popupContent.style.transform = 'translate(-50%, -50%) scale(1)';
            
            // Generate plots if needed
            if (properties.number_of_systems > 30) {
                this.generatePlots(properties, popupElement);
            }
        }, 10);
    }
    
    hideCityPopup() {
        if (this.currentCityPopup) {
            this.currentCityPopup.style.opacity = '0';
            const popupContent = this.currentCityPopup.querySelector('.city-popup');
            popupContent.style.transform = 'translate(-50%, -50%) scale(0.9)';
            
            setTimeout(() => {
                if (this.currentCityPopup && this.currentCityPopup.parentNode) {
                    this.currentCityPopup.parentNode.removeChild(this.currentCityPopup);
                }
                this.currentCityPopup = null;
            }, 200);
        }
    }
    
    generatePlots(properties, popupElement) {
        const name = properties.nom || 'Unknown';
        const sanitizedName = name.replace(/\s+/g, '-');
        
        // Generate sample data (replace with actual data when available)
        const sampleData = this.generateSampleData(properties.number_of_systems);
        
        // Create capacity distribution chart
        this.createCapacityChart(sanitizedName, sampleData.capacities);
        
        // Create tilt angle distribution chart
        this.createTiltChart(sanitizedName, sampleData.tilts);
        
        // Create azimuth chart
        this.createAzimuthChart(sanitizedName, sampleData.azimuths);
    }
    
    generateSampleData(numSystems) {
        // Generate realistic sample data for demonstration
        const capacities = [];
        const tilts = [];
        const azimuths = [];
        
        for (let i = 0; i < numSystems; i++) {
            // Capacity: mostly residential (3-10 kWp) with some commercial
            const capacity = Math.random() < 0.8 ? 
                3 + Math.random() * 7 : // Residential
                10 + Math.random() * 40; // Commercial
            capacities.push(capacity);
            
            // Tilt: mostly 20-40 degrees (typical for France)
            const tilt = 15 + Math.random() * 35;
            tilts.push(tilt);
            
            // Azimuth: mostly south-facing (135-225 degrees)
            const azimuth = 135 + Math.random() * 90;
            azimuths.push(azimuth);
        }
        
        return { capacities, tilts, azimuths };
    }
    
    createCapacityChart(cityName, capacities) {
        const ctx = document.getElementById(`capacity-chart-${cityName}`);
        if (!ctx) return;
        
        // Create histogram data
        const bins = this.createHistogram(capacities, 8);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: bins.labels,
                datasets: [{
                    label: 'Number of Systems',
                    data: bins.values,
                    backgroundColor: 'rgba(45, 55, 72, 0.8)',
                    borderColor: 'rgba(45, 55, 72, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Systems'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Capacity (kWp)'
                        }
                    }
                }
            }
        });
    }
    
    createTiltChart(cityName, tilts) {
        const ctx = document.getElementById(`tilt-chart-${cityName}`);
        if (!ctx) return;
        
        // Create histogram data
        const bins = this.createHistogram(tilts, 8);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: bins.labels,
                datasets: [{
                    label: 'Number of Systems',
                    data: bins.values,
                    backgroundColor: 'rgba(45, 55, 72, 0.8)',
                    borderColor: 'rgba(45, 55, 72, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Tilt Angle (°)'
                        }
                    }
                }
            }
        });
    }
    
    createAzimuthChart(cityName, azimuths) {
        const ctx = document.getElementById(`azimuth-chart-${cityName}`);
        if (!ctx) return;
        
        // Create polar histogram data for azimuth (0-360 degrees)
        const polarData = this.createPolarHistogram(azimuths, 12);
        
        const chart = new Chart(ctx, {
            type: 'polarArea',
            data: {
                labels: polarData.labels,
                datasets: [{
                    label: 'Number of Systems',
                    data: polarData.values,
                    backgroundColor: polarData.colors,
                    borderColor: 'rgba(45, 55, 72, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        ticks: {
                            display: false
                        },
                        grid: {
                            color: 'rgba(45, 55, 72, 0.1)'
                        }
                    }
                }
            }
        });
    }
    

    
    createHistogram(data, numBins) {
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binSize = (max - min) / numBins;
        
        const bins = new Array(numBins).fill(0);
        const labels = [];
        
        for (let i = 0; i < numBins; i++) {
            const binStart = min + i * binSize;
            const binEnd = min + (i + 1) * binSize;
            labels.push(`${binStart.toFixed(1)}-${binEnd.toFixed(1)}`);
        }
        
        data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), numBins - 1);
            bins[binIndex]++;
        });
        
        return { labels, values: bins };
    }
    
    createAzimuthHistogram(data, numBins) {
        const binSize = 360 / numBins;
        const bins = new Array(numBins).fill(0);
        const labels = [];
        
        // Create labels for azimuth bins (0-360 degrees)
        for (let i = 0; i < numBins; i++) {
            const binStart = i * binSize;
            const binEnd = (i + 1) * binSize;
            labels.push(`${binStart}-${binEnd}°`);
        }
        
        // Bin the azimuth data
        data.forEach(value => {
            const binIndex = Math.floor(value / binSize) % numBins;
            bins[binIndex]++;
        });
        
        return { labels, values: bins };
    }
    
    createPolarHistogram(data, numBins) {
        const binSize = 360 / numBins;
        const bins = new Array(numBins).fill(0);
        const labels = [];
        const colors = [];
    
        // Cardinal directions mapping (with 0° = top, 180° = bottom)
        const getDirectionLabel = (angle) => {
            if ((angle >= 337.5 && angle < 360) || (angle >= 0 && angle < 22.5)) return 'N (0°)';
            if (angle >= 67.5 && angle < 112.5) return 'E (90°)';
            if (angle >= 157.5 && angle < 202.5) return 'S (180°)';
            if (angle >= 247.5 && angle < 292.5) return 'W (270°)';
            return `${angle}°`;
        };
    
        // Create labels and colors for each bin
        for (let i = 0; i < numBins; i++) {
            const angle = (i * binSize) % 360;
            const directionLabel = getDirectionLabel(angle);
            labels.push(directionLabel);
            colors.push('rgba(45, 55, 72, 0.8)');
        }
    
        // Bin the azimuth data with proper orientation
        // In polar charts, 0° is at the top, so we need to rotate the data
        // so that 180° (south) appears at the bottom
        data.forEach(value => {
            // Rotate so that 180° (south) is at the bottom
            // Chart.js polar charts start at 0° (top) and go clockwise
            // We want 0° (north) at top, 90° (east) at right, 180° (south) at bottom, 270° (west) at left
            const adjustedValue = (360 - value) % 360;
            const binIndex = Math.floor(adjustedValue / binSize) % numBins;
            bins[binIndex]++;
        });
    
        return { labels, values: bins, colors };
    }
    

    
    searchCities(query) {
        if (query.length < 2) {
            this.clearSearch();
            return;
        }
        
        // Filter cities that match the query
        this.searchResults = this.allMarkers.filter(markerData => {
            const cityName = markerData.properties.nom || '';
            return cityName.toLowerCase().includes(query);
        });
        
        // Display search results
        this.displaySearchResults();
        
        // Highlight matching markers on map
        this.highlightSearchResults();
    }
    
    displaySearchResults() {
        const resultsContainer = document.getElementById('searchResults');
        
        if (this.searchResults.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">No cities found</div>';
            resultsContainer.style.display = 'block';
            return;
        }
        
        // Limit to first 10 results
        const limitedResults = this.searchResults.slice(0, 10);
        
        const resultsHTML = limitedResults.map(markerData => {
            const properties = markerData.properties;
            const name = properties.nom || 'Unknown';
            const capacity = properties.installed_capacity;
            const numSystems = properties.number_of_systems;
            
            // Check if there are no detections
            const hasDetections = capacity !== null && capacity !== undefined && capacity !== 'None' && 
                                 numSystems !== null && numSystems !== undefined && numSystems !== 'None';
            
            let statsText;
            if (hasDetections) {
                statsText = `${parseFloat(capacity).toFixed(1)} kWp • ${numSystems} systems`;
            } else {
                statsText = 'No detections';
            }
            
            return `
                <div class="search-result-item" onclick="window.deepPVMapperMap.focusOnCity('${name}')">
                    <div class="search-result-name">${name}</div>
                    <div class="search-result-stats">${statsText}</div>
                </div>
            `;
        }).join('');
        
        resultsContainer.innerHTML = resultsHTML;
        resultsContainer.style.display = 'block';
    }
    
    highlightSearchResults() {
        // Reset all markers to normal
        this.allMarkers.forEach(markerData => {
            markerData.marker.setIcon(L.divIcon({
                className: 'custom-marker',
                html: '<div class="marker-dot"></div>',
                iconSize: [8, 8],
                iconAnchor: [4, 4]
            }));
        });
        
        // Highlight search result markers
        this.searchResults.forEach(markerData => {
            markerData.marker.setIcon(L.divIcon({
                className: 'custom-marker search-highlight',
                html: '<div class="marker-dot search-highlight"></div>',
                iconSize: [12, 12],
                iconAnchor: [6, 6]
            }));
        });
    }
    
    focusOnCity(cityName) {
        const markerData = this.searchResults.find(m => m.properties.nom === cityName);
        if (markerData) {
            // Fly to the city
            this.map.flyTo(markerData.coordinates, 12, {
                duration: 1.5,
                easeLinearity: 0.25
            });
            
            // Show popup after animation
            setTimeout(() => {
                this.showCityPopup(markerData.properties, markerData.coordinates);
            }, 1500);
            
            // Clear search
            this.clearSearch();
        }
    }
    
    clearSearch() {
        this.searchResults = [];
        
        // Reset all markers to normal
        this.allMarkers.forEach(markerData => {
            markerData.marker.setIcon(L.divIcon({
                className: 'custom-marker',
                html: '<div class="marker-dot"></div>',
                iconSize: [8, 8],
                iconAnchor: [4, 4]
            }));
        });
        
        // Hide search results
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.style.display = 'none';
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