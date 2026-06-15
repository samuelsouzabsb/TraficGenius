"""
run_pipeline.py
===============
Orquestrador principal: processa todos os arquivos .pbf em sequência
(Brasil e EUA) e executa o join final.

Uso:
    python run_pipeline.py                    # processa tudo
    python run_pipeline.py --smoke-test       # só Delaware (validação)
    python run_pipeline.py --country BR       # só Brasil
    python run_pipeline.py --country US       # só EUA
    python run_pipeline.py --join-only        # só etapa 3 (join)
    python run_pipeline.py --skip-extraction  # pula etapa 1 (reusa parquets)
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS  = BASE_DIR / "scripts" / "infra"
INFRA_BR = BASE_DIR / "dataset" / "infra" / "brasil"
INFRA_US = BASE_DIR / "dataset" / "infra" / "us"
FEAT_DIR = BASE_DIR / "dataset" / "infra" / "h3_features"
ACCIDENTS= BASE_DIR / "dataset" / "dados_unificados.parquet"
OUTPUT   = BASE_DIR / "dataset" / "dataset_enriquecido.parquet"

EXTRACT_SCRIPT     = str(SCRIPTS / "01_extract_osm_features.py")
H3_SCRIPT          = str(SCRIPTS / "02_compute_h3_features.py")
CONSOLIDATE_SCRIPT = str(SCRIPTS / "03_consolidate_and_join.py")


def get_region_name(pbf_path: Path) -> str:
    stem = pbf_path.stem.replace("-260612", "")
    return stem.replace("-", "_")


def discover_pbf_files(country: str = "all") -> list:
    files = []
    if country in ("all", "BR"):
        for f in sorted(INFRA_BR.glob("*.pbf")):
            files.append((f, "br_" + get_region_name(f)))
    if country in ("all", "US"):
        for f in sorted(INFRA_US.glob("*.pbf")):
            files.append((f, "us_" + get_region_name(f)))
    return files


def run_step(cmd: list, label: str) -> bool:
    """Executa um subprocesso e retorna True se bem-sucedido."""
    print(f"\n    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERRO ERRO em: {label} (código {result.returncode})")
        return False
    return True


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def run_pipeline(
    smoke_test: bool = False,
    country: str = "all",
    join_only: bool = False,
    skip_extraction: bool = False,
):
    start_total = time.time()

    # Descobrir arquivos
    if smoke_test:
        smoke_path = INFRA_US / "delaware-260612.osm.pbf"
        if not smoke_path.exists():
            print(f"ERRO: arquivo de smoke test não encontrado: {smoke_path}")
            sys.exit(1)
        files = [(smoke_path, "us_delaware")]
        print(f"\n{'='*70}")
        print("  MODO SMOKE TEST — Delaware apenas")
        print(f"{'='*70}")
    else:
        files = discover_pbf_files(country)
        print(f"\n{'='*70}")
        print(f"  PIPELINE COMPLETO — {len(files)} arquivos PBF | País: {country}")
        print(f"{'='*70}")

    errors = []

    if not join_only:
        total = len(files)
        for i, (pbf_path, region) in enumerate(files, 1):
            print(f"\n{'-'*70}")
            print(f"  [{i}/{total}] {region}")
            print(f"  Arquivo: {pbf_path.name} ({pbf_path.stat().st_size/1024/1024:.0f} MB)")
            print(f"{'-'*70}")
            t0 = time.time()

            # ETAPA 1 — Extração
            if not skip_extraction:
                ok = run_step(
                    [sys.executable, EXTRACT_SCRIPT, str(pbf_path), "--region", region],
                    f"Extração OSM {region}"
                )
                if not ok:
                    errors.append((region, "extração"))
                    continue

            # ETAPA 2 — Agregação H3
            ok = run_step(
                [sys.executable, H3_SCRIPT, "--region", region],
                f"Agregação H3 {region}"
            )
            if not ok:
                errors.append((region, "agregação H3"))
                continue

            elapsed = time.time() - t0
            print(f"\n  OK {region} concluído em {fmt_time(elapsed)}")

    # ETAPA 3 — Consolidação e Join
    print(f"\n{'='*70}")
    print("  ETAPA 3 — Consolidação Global e Join Final")
    print(f"{'='*70}")

    if smoke_test:
        out_arg = str(BASE_DIR / "dataset" / "dataset_enriquecido_smoke.parquet")
    else:
        out_arg = str(OUTPUT)

    ok = run_step(
        [sys.executable, CONSOLIDATE_SCRIPT,
         "--accidents", str(ACCIDENTS),
         "--output", out_arg],
        "Consolidação e Join"
    )
    if not ok:
        errors.append(("global", "consolidação/join"))

    # Sumário final
    total_elapsed = time.time() - start_total
    print(f"\n{'='*70}")
    if errors:
        print(f"  AVISO  PIPELINE FINALIZADO COM {len(errors)} ERRO(S):")
        for region, step in errors:
            print(f"     - {region}: falhou na {step}")
    else:
        print(f"  OK  PIPELINE CONCLUÍDO SEM ERROS")

    print(f"  Tempo total: {fmt_time(total_elapsed)}")
    print(f"  Saída: {out_arg}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline OSM → H3 Features → Join de Acidentes")
    parser.add_argument("--smoke-test",      action="store_true", help="Processar apenas Delaware (validação)")
    parser.add_argument("--country",         default="all",       choices=["all", "BR", "US"],
                        help="Filtrar por país")
    parser.add_argument("--join-only",       action="store_true", help="Apenas etapa 3 (parquets já existentes)")
    parser.add_argument("--skip-extraction", action="store_true", help="Pular extração OSM (reusa parquets)")
    args = parser.parse_args()

    run_pipeline(
        smoke_test=args.smoke_test,
        country=args.country,
        join_only=args.join_only,
        skip_extraction=args.skip_extraction,
    )
