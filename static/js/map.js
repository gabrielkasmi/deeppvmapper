// Popup functions
function openPopup() {
    document.getElementById('infoPopup').style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closePopup() {
    document.getElementById('infoPopup').style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Initialize popup - hide by default
document.addEventListener('DOMContentLoaded', function() {
    // Hide popup by default
    document.getElementById('infoPopup').style.display = 'none';
    
    // Close popup when clicking outside
    const popup = document.getElementById('infoPopup');
    popup.addEventListener('click', function(e) {
        if (e.target === popup) {
            closePopup();
        }
    });
});

// DeepPVMapper Interactive Map
class DeepPVMapperMap {
    constructor() {
        this.map = null;
        this.pvLayer = null;
        this.currentData = [];
        this.filters = {
            region: '',
            powerRange: '',
            year: ''
        };
        
        this.init();
    }
    
    init() {
        this.initMap();
        this.initFilters();
        this.loadSampleData();
        this.updateStats();
    }
    
    initMap() {
        // Initialize the map centered on France
        this.map = L.map('map').setView([46.603354, 1.888334], 6);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }).addTo(this.map);
        
        // Create a layer group for PV installations
        this.pvLayer = L.layerGroup().addTo(this.map);
    }
    
    initFilters() {
        // Region filter
        document.getElementById('region-select').addEventListener('change', (e) => {
            this.filters.region = e.target.value;
            this.applyFilters();
        });
        
        // Power range filter
        document.getElementById('power-range').addEventListener('change', (e) => {
            this.filters.powerRange = e.target.value;
            this.applyFilters();
        });
        
        // Year filter
        document.getElementById('year-filter').addEventListener('change', (e) => {
            this.filters.year = e.target.value;
            this.applyFilters();
        });
    }
    
    loadSampleData() {
        // This is placeholder data - replace with actual GeoJSON loading
        const sampleData = [
            {
                type: "Feature",
                geometry: {
                    type: "Polygon",
                    coordinates: [[[2.3522, 48.8566], [2.3522, 48.8567], [2.3523, 48.8567], [2.3523, 48.8566], [2.3522, 48.8566]]]
                },
                properties: {
                    id: "PV_001",
                    power: 5.2,
                    year: 2022,
                    region: "ile-de-france",
                    tilt: 30,
                    orientation: 180,
                    surface: 35.2
                }
            },
            {
                type: "Feature",
                geometry: {
                    type: "Polygon",
                    coordinates: [[[4.8357, 45.7640], [4.8357, 45.7641], [4.8358, 45.7641], [4.8358, 45.7640], [4.8357, 45.7640]]]
                },
                properties: {
                    id: "PV_002",
                    power: 12.8,
                    year: 2023,
                    region: "auvergne-rhone-alpes",
                    tilt: 25,
                    orientation: 165,
                    surface: 85.6
                }
            }
        ];
        
        this.currentData = sampleData;
        this.displayData(sampleData);
    }
    
    displayData(data) {
        // Clear existing markers
        this.pvLayer.clearLayers();
        
        data.forEach(feature => {
            const properties = feature.properties;
            const power = properties.power;
            
            // Determine color based on power range
            let color = '#ff6b6b'; // 0-3 kWc
            if (power > 9) {
                color = '#45b7d1'; // 9-36 kWc
            } else if (power > 3) {
                color = '#4ecdc4'; // 3-9 kWc
            }
            
            // Create polygon
            const polygon = L.polygon(feature.geometry.coordinates, {
                color: color,
                weight: 2,
                fillColor: color,
                fillOpacity: 0.6
            });
            
            // Create popup content
            const popupContent = `
                <div class="pv-popup">
                    <h4>PV Installation ${properties.id}</h4>
                    <p><strong>Power:</strong> ${properties.power} kWc</p>
                    <p><strong>Year:</strong> ${properties.year}</p>
                    <p><strong>Region:</strong> ${this.getRegionName(properties.region)}</p>
                    <p><strong>Tilt:</strong> ${properties.tilt}°</p>
                    <p><strong>Orientation:</strong> ${properties.orientation}°</p>
                    <p><strong>Surface:</strong> ${properties.surface} m²</p>
                </div>
            `;
            
            polygon.bindPopup(popupContent);
            polygon.addTo(this.pvLayer);
        });
    }
    
    applyFilters() {
        let filteredData = this.currentData;
        
        // Apply region filter
        if (this.filters.region) {
            filteredData = filteredData.filter(item => 
                item.properties.region === this.filters.region
            );
        }
        
        // Apply power range filter
        if (this.filters.powerRange) {
            const [min, max] = this.filters.powerRange.split('-').map(Number);
            filteredData = filteredData.filter(item => {
                const power = item.properties.power;
                if (max) {
                    return power >= min && power < max;
                } else {
                    return power >= min;
                }
            });
        }
        
        // Apply year filter
        if (this.filters.year) {
            filteredData = filteredData.filter(item => 
                item.properties.year === parseInt(this.filters.year)
            );
        }
        
        this.displayData(filteredData);
        this.updateStats(filteredData);
    }
    
    updateStats(data = this.currentData) {
        const totalInstallations = data.length;
        const totalPower = data.reduce((sum, item) => sum + item.properties.power, 0);
        const avgPower = totalInstallations > 0 ? totalPower / totalInstallations : 0;
        
        document.getElementById('total-installations').textContent = totalInstallations;
        document.getElementById('total-power').textContent = (totalPower / 1000).toFixed(1);
        document.getElementById('avg-power').textContent = avgPower.toFixed(1);
    }
    
    getRegionName(regionCode) {
        const regions = {
            'ile-de-france': 'Île-de-France',
            'provence-alpes-cote-azur': 'Provence-Alpes-Côte d\'Azur',
            'occitanie': 'Occitanie',
            'nouvelle-aquitaine': 'Nouvelle-Aquitaine',
            'auvergne-rhone-alpes': 'Auvergne-Rhône-Alpes',
            'hauts-de-france': 'Hauts-de-France',
            'grand-est': 'Grand Est',
            'bourgogne-franche-comte': 'Bourgogne-Franche-Comté',
            'centre-val-de-loire': 'Centre-Val de Loire',
            'normandie': 'Normandie',
            'bretagne': 'Bretagne',
            'pays-de-la-loire': 'Pays de la Loire'
        };
        return regions[regionCode] || regionCode;
    }
    
    // Method to load actual GeoJSON data
    loadGeoJSONData(url) {
        fetch(url)
            .then(response => response.json())
            .then(data => {
                this.currentData = data.features || data;
                this.displayData(this.currentData);
                this.updateStats();
            })
            .catch(error => {
                console.error('Error loading GeoJSON data:', error);
            });
    }
}

// Initialize the map when the page loads
document.addEventListener('DOMContentLoaded', function() {
    new DeepPVMapperMap();
}); 