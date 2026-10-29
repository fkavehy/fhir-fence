# Data

This repository does **not** redistribute the dataset.

## Obtain the dataset
Download the open-access **MIMIC-IV Clinical Database Demo on FHIR v2.1.0** from PhysioNet:
https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/

- Access: Open — no credentialing required.
- License: Open Data Commons Open Database License (ODbL) v1.0.
- The data are real **de-identified** patient records (not synthetic patients).

Point `configs/config.yaml → dataset.base_url` at the `fhir/` directory, or place the
`*.ndjson.gz` files under `data/raw/mimic-iv-fhir-demo-2.1.0/`.

## Citations (verify against the source before use)
- Bennett A, Ulrich H, Wiedekopf J, Szul P, Grimes J, Johnson A. MIMIC-IV Clinical
  Database Demo on FHIR (v2.1.0). PhysioNet, 2025.
- Bennett AM, Ulrich H, van Damme P, Wiedekopf J, Johnson AEW. MIMIC-IV on FHIR:
  converting a decade of in-patient data into an exchangeable, interoperable format.
  JAMIA. 2023;30(4):718-725. doi:10.1093/jamia/ocad002
- Parent projects: MIMIC-IV Clinical Database Demo and MIMIC-IV-ED Demo.

## Secrets
All values treated as secrets in this project are **synthetic canaries** generated
deterministically by `src/fhirfence/graph.py`. The canary registry is derived from a
seed and is never committed. It must never be included in any model prompt.
