import os
import urllib.request

def download_file(url, dest):
    print(f"Baixando {url} para {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Sucesso!")
    except Exception as e:
        print(f"Erro ao baixar: {e}")

if __name__ == "__main__":
    libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
    os.makedirs(libs_dir, exist_ok=True)
    
    urls = {
        "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
        "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
        "chart.js": "https://cdn.jsdelivr.net/npm/chart.js",
        "chartjs-plugin-datalabels.js": "https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"
    }
    
    for filename, url in urls.items():
        dest_path = os.path.join(libs_dir, filename)
        download_file(url, dest_path)
