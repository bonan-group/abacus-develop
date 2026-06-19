#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fail(message):
    print(f"validate_training_dump: {message}", file=sys.stderr)
    return 1


def load_array(dump_dir, arrays, key):
    filename = arrays.get(key)
    if filename is None:
        raise KeyError(f'missing arrays["{key}"]')
    path = dump_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing file for {key}: {path}")
    data = np.load(path, allow_pickle=False)
    if not np.isfinite(data).all():
        raise ValueError(f"{key} contains non-finite values")
    return data


def main():
    parser = argparse.ArgumentParser(description="Validate an ABACUS PW training_dump/step_* directory.")
    parser.add_argument("dump_dir", help="Path to a training_dump step directory containing record.json")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    record_path = dump_dir / "record.json"
    if not record_path.is_file():
        return fail(f"record.json not found: {record_path}")

    with record_path.open() as fh:
        record = json.load(fh)

    arrays = record.get("arrays")
    if not isinstance(arrays, dict):
        return fail('record.json missing object field "arrays"')

    try:
        nspin = int(record["nspin"])
        nsigma = int(record["nsigma"])
        nxyz = int(record["nxyz"])
        rho_valence = load_array(dump_dir, arrays, "rho_valence_sg")
        rho = load_array(dump_dir, arrays, "rho_sg")
        rho_core = load_array(dump_dir, arrays, "rho_core_g")
        sigma = load_array(dump_dir, arrays, "sigma_xg")
        sigma_valence = load_array(dump_dir, arrays, "sigma_valence_xg")
    except (KeyError, FileNotFoundError, ValueError, TypeError) as exc:
        return fail(str(exc))

    expected_rho_shape = (nspin, nxyz)
    expected_sigma_shape = (nsigma, nxyz)
    expected_core_shape = (nxyz,)
    checks = [
        ("rho_valence_sg", rho_valence.shape, expected_rho_shape),
        ("rho_sg", rho.shape, expected_rho_shape),
        ("rho_core_g", rho_core.shape, expected_core_shape),
        ("sigma_xg", sigma.shape, expected_sigma_shape),
        ("sigma_valence_xg", sigma_valence.shape, expected_sigma_shape),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            return fail(f"{name} has shape {actual}, expected {expected}")

    if not np.allclose(rho, rho_valence + rho_core[None, :] / nspin, rtol=1.0e-10, atol=1.0e-12):
        return fail("rho_sg is not rho_valence_sg + rho_core_g / nspin")

    tau_present = bool(record.get("tau_present"))
    if tau_present:
        try:
            tau = load_array(dump_dir, arrays, "tau_sg")
            tau_valence = load_array(dump_dir, arrays, "tau_valence_sg")
        except (KeyError, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        for name, data in [("tau_sg", tau), ("tau_valence_sg", tau_valence)]:
            if data.shape != expected_rho_shape:
                return fail(f"{name} has shape {data.shape}, expected {expected_rho_shape}")
    elif "tau_valence_sg" in arrays:
        return fail('arrays["tau_valence_sg"] present while tau_present is false')

    print(f"validated {dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
