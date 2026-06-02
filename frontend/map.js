/**
 * @file map.js
 * @description Script de controle do mapa interativo por meio da biblioteca Leaflet.
 * Desenha marcadores hexagonais vetoriais brilhantes com filtros de brilho (glow effect)
 * integrando dados geográficos à estética geral do painel.
 * 
 * Dicas de Inglês (English Tips):
 * - 'Tile layer' refere-se às camadas de blocos/imagens que formam o plano de fundo do mapa (ex: CartoDB Dark).
 * - 'Attribution control' é o controle de créditos das fontes de dados geográficos no rodapé do mapa.
 * - 'Resize Observer' é uma API nativa do navegador que observa mudanças dimensionais físicas em elementos HTML.
 * - 'Anchor' refere-se ao ponto de ancoragem que fixa o ícone no local exato da coordenada geográfica.
 * - 'SVG filter' é o filtro de efeitos gráficos definidos em XML de vetor para criar sombras, desfoques e brilhos.
 * - 'Marker' é o marcador/pin colocado sobre uma coordenada específica do mapa.
 */

let myMap; // Variável global para armazenar a instância do mapa Leaflet

/**
 * Inicializa a projeção geográfica do mapa centrado na América do Norte.
 * Configura o plano de fundo escuro e o redimensionador responsivo automático.
 */
function initMap() {
    // Inicializa o mapa focado nos EUA com coordenadas de latitude/longitude e zoom inicial 5
    myMap = L.map('map', {
        zoomControl: false,        // Desabilita os botões +/- do zoom para manter o visual HUD desobstruído
        attributionControl: false  // Oculta a caixa de créditos para manter o design cinematográfico
    }).setView([39.8283, -98.5795], 5);

    // Adiciona a camada de mapa escuro minimalista da CartoDB (Dark Matter tile layer)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd'
    }).addTo(myMap);

    // Ajusta o tamanho da área do mapa no ciclo inicial após a tela carregar
    setTimeout(() => {
        myMap.invalidateSize();
    }, 500);

    // Cria um observador (ResizeObserver) para ajustar as proporções internas do Leaflet
    // se o elemento do mapa for redimensionado fisicamente no navegador (responsividade)
    const mapContainer = document.getElementById('map');
    if (mapContainer && typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => {
            myMap.invalidateSize(); // Corrige renderizações cortadas ou cinzas
        }).observe(mapContainer);
    }
}

/**
 * Limpa marcadores anteriores e desenha os hexágonos brilhantes dinâmicos.
 * 
 * Parâmetros (Parameters):
 * - clusters (array): Vetor de objetos contendo latitude, longitude e severidade de cada ponto.
 */
function plotClusters(clusters) {
    // Garante que o mapa esteja inicializado antes de plotar os pontos
    if (!myMap) initMap();
    
    // Varre as camadas ativas e remove marcadores, círculos ou polígonos legados
    myMap.eachLayer((layer) => {
        if (layer instanceof L.Marker || layer instanceof L.Circle || layer instanceof L.CircleMarker) {
            myMap.removeLayer(layer);
        }
    });

    // Loop para desenhar cada ponto de acidente
    clusters.forEach(cluster => {
        // Obtém a cor neon ativa configurada no CSS dinamicamente usando getComputedStyle
        const mainColor = getComputedStyle(document.documentElement).getPropertyValue('--neon-cyan').trim() || '#00e5ff';
        const isHighSeverity = cluster.severity > 3;  // Flag para acidentes críticos

        // Código SVG Inline personalizado: desenha um hexágono externo brilhante com o tema de cor ativo,
        // um hexágono interno de calor (Vermelho/Rosa para grave, Laranja para moderado) e um ponto branco central.
        // Utiliza feGaussianBlur para criar o brilho reflexivo (glow effect).
        const svgIcon = `
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="glow-${cluster.severity}">
                    <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <!-- Hexágono Externo (Neon Theme Color) -->
            <polygon points="50,5 90,25 90,75 50,95 10,75 10,25" fill="${mainColor}25" stroke="${mainColor}" stroke-width="1.5" filter="url(#glow-${cluster.severity})"/>
            <!-- Hexágono Interno de Risco (Vermelho ou Laranja) -->
            <polygon points="50,25 75,40 75,60 50,75 25,60 25,40" fill="${isHighSeverity ? 'rgba(255, 42, 85, 0.5)' : 'rgba(255, 138, 0, 0.5)'}" stroke="${isHighSeverity ? '#FF2A55' : '#FF8A00'}" stroke-width="2" filter="url(#glow-${cluster.severity})"/>
            <!-- Ponto Centralizador -->
            <circle cx="50" cy="50" r="6" fill="#FFF" />
        </svg>
        `;

        // Define a dimensão física do hexágono no mapa: maior para gravidade crítica
        const size = cluster.severity > 3 ? 80 : 60;

        // Converte o código SVG bruto em um ícone HTML reconhecido pelo Leaflet (DivIcon)
        const icon = L.divIcon({
            html: svgIcon,
            className: 'hex-marker',
            iconSize: [size, size],
            iconAnchor: [size/2, size/2] // Centraliza a ancoragem no meio do SVG
        });

        // Instancia o marcador geográfico, adiciona ao mapa e configura o balão pop-up de detalhes
        L.marker([cluster.lat, cluster.lng], { icon: icon }).addTo(myMap)
            .bindPopup(`
                <div style="background: #05050A; padding: 10px; border-radius: 6px; border: 1px solid #FF2A55; color: #FFF;">
                    <b>Severity Level:</b> <span style="color: #FF2A55; font-weight: bold; font-size: 1.2rem;">${cluster.severity}</span>
                </div>
            `, {
                className: 'custom-dark-popup'
            });
    });
}
