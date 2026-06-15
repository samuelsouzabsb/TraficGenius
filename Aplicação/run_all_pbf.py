"""
run_all_pbf.py
==============
Processa todos os arquivos PBF do Brasil e EUA em sequencia.
Usa resumability - pula arquivos ja processados.
"""
import sys
import os
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS  = BASE_DIR / "scripts" / "infra"
INFRA_BR = BASE_DIR / "dataset" / "infra" / "brasil"
INFRA_US = BASE_DIR / "dataset" / "infra" / "us"
PROC_DIR = BASE_DIR / "dataset" / "infra" / "processed"

EXTRACT_SCRIPT = str(SCRIPTS / "01_extract_osm_features.py")
H3_SCRIPT      = str(SCRIPTS / "02_compute_h3_features.py")

LOG_FILE = BASE_DIR / "run_all_pbf.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_region(pbf_path: Path, prefix: str) -> str:
    stem = pbf_path.stem.replace("-260612", "").replace("-", "_")
    return f"{prefix}_{stem}"

def already_done(region: str) -> bool:
    h11 = BASE_DIR / "dataset" / "infra" / "h3_features" / "h3_11" / f"h3_11_{region}.parquet"
    h10 = BASE_DIR / "dataset" / "infra" / "h3_features" / "h3_10" / f"h3_10_{region}.parquet"
    h9  = BASE_DIR / "dataset" / "infra" / "h3_features" / "h3_9"  / f"h3_9_{region}.parquet"
    return h11.exists() and h10.exists() and h9.exists()

def run(cmd, label):
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode not in (0, 1):  # 1 = unicode stderr warning, still OK
        log(f"  ERRO CRITICO em {label}: codigo {result.returncode}")
        return False
    return True

def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}min"
    return f"{s/3600:.1f}h"

def process_file(pbf_path: Path, region: str, idx: int, total: int):
    size_mb = pbf_path.stat().st_size / 1024 / 1024
    log(f"[{idx}/{total}] {region} ({size_mb:.0f} MB)")

    if already_done(region):
        log(f"  SKIP - ja processado")
        return True

    t0 = time.time()

    # Verificar se extracao ja existe
    nodes_ok = (PROC_DIR / f"nodes_{region}.parquet").exists()
    ways_ok  = (PROC_DIR / f"ways_{region}.parquet").exists()
    land_ok  = (PROC_DIR / f"landuse_{region}.parquet").exists()

    if nodes_ok and ways_ok and land_ok:
        log(f"  Extracao ja existe, pulando para agregacao H3 ...")
    else:
        log(f"  Iniciando extracao OSM ...")
        ok = run([sys.executable, "-X", "utf8", EXTRACT_SCRIPT, str(pbf_path), "--region", region],
                 f"extracao {region}")
        if not ok:
            log(f"  FALHOU na extracao de {region}")
            return False
        log(f"  Extracao concluida em {fmt_time(time.time()-t0)}")

    # Agregacao H3
    t1 = time.time()
    log(f"  Iniciando agregacao H3 ...")
    ok = run([sys.executable, "-X", "utf8", H3_SCRIPT, "--region", region],
             f"agregacao H3 {region}")
    if not ok:
        log(f"  FALHOU na agregacao H3 de {region}")
        return False

    total_time = time.time() - t0
    log(f"  OK {region} concluido em {fmt_time(total_time)}")
    return True

def main():
    log("="*60)
    log("INICIO DO PROCESSAMENTO COMPLETO DE PBFs")
    log("="*60)

    # Montar lista de todos os arquivos
    files = []
    for pbf in sorted(INFRA_BR.glob("*.pbf")):
        files.append((pbf, get_region(pbf, "br")))
    for pbf in sorted(INFRA_US.glob("*.pbf")):
        files.append((pbf, get_region(pbf, "us")))

    log(f"Total: {len(files)} arquivos PBF")

    errors = []
    total = len(files)
    start = time.time()

    for i, (pbf, region) in enumerate(files, 1):
        ok = process_file(pbf, region, i, total)
        if not ok:
            errors.append(region)

        # Estimar tempo restante
        elapsed = time.time() - start
        done_count = i
        if done_count > 0:
            avg = elapsed / done_count
            remaining = avg * (total - done_count)
            log(f"  Progresso: {done_count}/{total} | Tempo restante estimado: {fmt_time(remaining)}")

    log("="*60)
    if errors:
        log(f"CONCLUIDO COM {len(errors)} ERROS:")
        for e in errors:
            log(f"  - {e}")
    else:
        log("TODOS OS PBFS PROCESSADOS COM SUCESSO!")
    log(f"Tempo total: {fmt_time(time.time()-start)}")
    log("="*60)
    log("Proximos passos: executar 03_consolidate_and_join.py")

if __name__ == "__main__":
    main()
