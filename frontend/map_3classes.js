/**
 * @file map_3classes.js
 * @description Script de controle do mapa interativo (3 Classes) por meio da biblioteca Leaflet.
 */

let myMap;

function initMap() {
    myMap = L.map('map', {
        zoomControl: false,
        attributionControl: false
    }).setView([39.8283, -98.5795], 5);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd'
    }).addTo(myMap);

    setTimeout(() => {
        myMap.invalidateSize();
    }, 500);

    const mapContainer = document.getElementById('map');
    if (mapContainer && typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => {
            myMap.invalidateSize();
        }).observe(mapContainer);
    }
}

function getSeverityLabel(sevVal) {
    const s = Math.round(sevVal);
    if (s <= 1) return "Leve/Médio";
    if (s === 2) return "Grave";
    return "Fatal";
}

function plotClusters(points, isMacro = true) {
    if (!myMap) initMap();
    
    myMap.eachLayer((layer) => {
        if (layer instanceof L.Marker || layer instanceof L.Circle || layer instanceof L.CircleMarker) {
            myMap.removeLayer(layer);
        }
    });

    const selectAccident = document.getElementById('select-accident-id');
    if (selectAccident) {
        if (!isMacro) {
            selectAccident.innerHTML = '<option value="">Selecione pelo ID...</option>';
            points.forEach(point => {
                const opt = document.createElement('option');
                opt.value = point.id;
                opt.textContent = `ID #${point.id} (${getSeverityLabel(point.Severidade)})`;
                selectAccident.appendChild(opt);
            });
        }
    }

    points.forEach(point => {
        const lat = isMacro ? point.lat : point.Latitude_Inicial;
        const lng = isMacro ? point.lng : point.Longitude_Inicial;
        const severity = isMacro ? point.severity : point.Severidade;
        const cluster_id = isMacro ? point.cluster_id : point.Cluster_Espacial;
        const accident_id = isMacro ? null : point.id;

        const mainColor = getComputedStyle(document.documentElement).getPropertyValue('--neon-cyan').trim() || '#00e5ff';
        const roundedSeverity = Math.round(severity) || 1;

        // Atribuição de cores para a escala de 3 classes
        let innerFill = 'rgba(0, 240, 255, 0.5)'; // Leve/Médio: Ciano
        let innerStroke = '#00F0FF';
        
        if (roundedSeverity === 2) {
            innerFill = 'rgba(255, 138, 0, 0.5)'; // Grave: Laranja
            innerStroke = '#FF8A00';
        } else if (roundedSeverity >= 3) {
            innerFill = 'rgba(255, 42, 85, 0.5)'; // Fatal: Vermelho
            innerStroke = '#FF2A55';
        }

        const svgIcon = `
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="glow-${roundedSeverity}">
                    <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <!-- Hexágono Externo -->
            <polygon points="50,5 90,25 90,75 50,95 10,75 10,25" fill="${mainColor}25" stroke="${mainColor}" stroke-width="1.5" filter="url(#glow-${roundedSeverity})"/>
            <!-- Hexágono Interno de Risco -->
            <polygon points="50,25 75,40 75,60 50,75 25,60 25,40" fill="${innerFill}" stroke="${innerStroke}" stroke-width="2" filter="url(#glow-${roundedSeverity})"/>
            <!-- Ponto Centralizador -->
            <circle cx="50" cy="50" r="6" fill="#FFF" />
        </svg>
        `;

        // Define a dimensão física do hexágono no mapa
        let size = 60;
        if (roundedSeverity === 2) size = 70;
        if (roundedSeverity >= 3) size = 80;

        const icon = L.divIcon({
            html: svgIcon,
            className: 'hex-marker',
            iconSize: [size, size],
            iconAnchor: [size/2, size/2]
        });

        const marker = L.marker([lat, lng], { icon: icon }).addTo(myMap);

        if (isMacro) {
            const displaySeverity = typeof severity === 'number' ? severity.toFixed(2) : severity;
            marker.bindPopup(`
                <div style="background: #05050A; padding: 10px; border-radius: 6px; border: 1px solid var(--neon-cyan); color: #FFF; font-family: 'Inter', sans-serif;">
                    <b style="color: var(--neon-cyan); font-size: 1.05rem;">Região Cluster ${cluster_id}</b><br/>
                    <div style="margin-top: 5px;">Média Severidade (3C): <span style="color: #FF8A00; font-weight: bold;">${displaySeverity}</span></div>
                    <div style="margin-top: 5px; font-size: 0.8rem; color: rgba(255,255,255,0.6);">Clique no marcador para explorar os acidentes desta região</div>
                </div>
            `, {
                className: 'custom-dark-popup'
            });

            marker.on('click', async () => {
                myMap.setView([lat, lng], 11);
                
                const backBtn = document.getElementById('btn-back-macro');
                if (backBtn) backBtn.style.display = 'flex';
                
                const titleEl = document.querySelector('.overlay-title');
                const subtitleEl = document.querySelector('.overlay-subtitle');
                if (titleEl) titleEl.textContent = `CLUSTER ${cluster_id} DETAILS`;
                if (subtitleEl) subtitleEl.textContent = `Visualizando acidentes (3 Classes) no Cluster ${cluster_id}`;

                try {
                    const res = await fetch(`http://127.0.0.1:8000/api/acidentes/?limit=100&cluster_id=${cluster_id}&classes=3`);
                    if (!res.ok) throw new Error("Falha ao buscar acidentes");
                    const data = await res.json();
                    plotClusters(data.results, false);
                } catch (err) {
                    console.error(err);
                }
            });
        } else {
            marker.bindPopup(`
                <div style="background: #05050A; padding: 10px; border-radius: 6px; border: 1px solid #FF2A55; color: #FFF; font-family: 'Inter', sans-serif;">
                    <b style="color: #FF2A55; font-size: 1.05rem;">Acidente #${accident_id}</b><br/>
                    <div style="margin-top: 5px;">Severidade: <span style="color: #FF2A55; font-weight: bold;">${getSeverityLabel(severity)}</span></div>
                    <div style="margin-top: 5px; font-size: 0.8rem; color: rgba(255,255,255,0.6);">Clique para carregar SHAP e detalhes do clima</div>
                </div>
            `, {
                className: 'custom-dark-popup'
            });

            marker.on('click', async () => {
                try {
                    let detail;
                    if (accident_id >= 9990) {
                        detail = window.defaultAccidentsData.find(d => d.id == accident_id);
                    } else {
                        const res = await fetch(`http://127.0.0.1:8000/api/acidentes/${accident_id}/?classes=3`);
                        if (!res.ok) throw new Error("Falha ao buscar detalhes do acidente");
                        detail = await res.json();
                    }

                    if (detail && typeof window.updateSelectedAccidentDetails === 'function') {
                        window.updateSelectedAccidentDetails(detail);
                    }
                } catch (err) {
                    console.error(err);
                }
            });
        }
    });
}
