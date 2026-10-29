"""Generate a tiny, fully SYNTHETIC FHIR dataset in MIMIC-IV-FHIR profile format.

Nothing here is real: names, dates, values and identifiers are invented. The output is
written as NDJSON under examples/synthetic_fhir/ in exactly the format fhirfence.data
expects, so the demo and tests run with no credentialed data and no network.

    python examples/make_synthetic_fhir.py            # writes examples/synthetic_fhir/
    python examples/make_synthetic_fhir.py --n 6 OUT  # 6 patients into OUT/

This is demo scaffolding, not part of the method.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROFILE = "http://mimic.mit.edu/fhir/mimic/StructureDefinition/"

CONDITIONS = ["Type 2 diabetes mellitus", "Essential hypertension", "Acute kidney injury",
              "Community-acquired pneumonia", "Atrial fibrillation", "Sepsis",
              "Chronic obstructive pulmonary disease", "Congestive heart failure"]
MEDS = ["Metformin 500 mg oral tablet", "Lisinopril 10 mg oral tablet",
        "Furosemide 40 mg injection", "Piperacillin-tazobactam 4.5 g injection",
        "Apixaban 5 mg oral tablet", "Insulin glargine 100 units/mL"]
LABS = [("Glucose", "mg/dL", (80, 240)), ("Potassium", "mmol/L", (3.2, 5.4)),
        ("Creatinine", "mg/dL", (0.6, 3.1)), ("Hemoglobin", "g/dL", (8.0, 15.0)),
        ("Sodium", "mmol/L", (132, 145)), ("Heart rate", "beats/min", (60, 120))]


def _res(rtype, rid, profile, **extra):
    r = {"resourceType": rtype, "id": rid,
         "meta": {"profile": [PROFILE + profile]}}
    r.update(extra)
    return r


def _cc(text):
    return {"coding": [{"display": text}], "text": text}


def generate(n_patients: int, out_dir: Path, seed: int = 424242):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list] = {}

    def add(profile, res):
        buckets.setdefault(profile, []).append(res)

    for i in range(n_patients):
        pk = f"synthetic-patient-{i:03d}"
        add("MimicPatient", _res(
            "Patient", pk, "MimicPatient",
            name=[{"use": "official", "family": "REALNAME-WILL-BE-OVERWRITTEN"}],
            gender=rng.choice(["male", "female"]),
            birthDate=f"{rng.randint(1945, 1990)}-0{rng.randint(1,9)}-1{rng.randint(0,9)}",
            maritalStatus=_cc(rng.choice(["Married", "Single", "Widowed"]))))

        enc_id = f"enc-{i:03d}-focal"
        start = f"21{rng.randint(10,30)}-0{rng.randint(1,9)}-1{rng.randint(0,9)}"
        add("MimicEncounter", _res(
            "Encounter", enc_id, "MimicEncounter",
            subject={"reference": f"Patient/{pk}"},
            **{"class": {"coding": [{"display": "inpatient"}]}},
            serviceType=_cc("General Medicine"),
            period={"start": f"{start}T08:00:00Z", "end": f"{start}T20:00:00Z"}))

        # a second, non-focal encounter so minimization has something to exclude
        enc2 = f"enc-{i:03d}-prior"
        add("MimicEncounter", _res(
            "Encounter", enc2, "MimicEncounter",
            subject={"reference": f"Patient/{pk}"},
            **{"class": {"coding": [{"display": "ambulatory"}]}},
            period={"start": "2105-01-01T09:00:00Z"}))

        for j, cond in enumerate(rng.sample(CONDITIONS, rng.randint(2, 4))):
            add("MimicCondition", _res(
                "Condition", f"cond-{i:03d}-{j}", "MimicCondition",
                subject={"reference": f"Patient/{pk}"},
                encounter={"reference": f"Encounter/{enc_id}"},
                code=_cc(cond), recordedDate=f"{start}T10:00:00Z"))
        # one condition on the OTHER encounter — must NOT appear in the T2 answer
        add("MimicCondition", _res(
            "Condition", f"cond-{i:03d}-prior", "MimicCondition",
            subject={"reference": f"Patient/{pk}"},
            encounter={"reference": f"Encounter/{enc2}"},
            code=_cc("Seasonal allergic rhinitis"), recordedDate="2105-01-01T10:00:00Z"))

        for j, med in enumerate(rng.sample(MEDS, rng.randint(2, 4))):
            add("MimicMedicationRequest", _res(
                "MedicationRequest", f"medreq-{i:03d}-{j}", "MimicMedicationRequest",
                subject={"reference": f"Patient/{pk}"},
                encounter={"reference": f"Encounter/{enc_id}"},
                medicationCodeableConcept=_cc(med),
                authoredOn=f"{start}T11:0{j}:00Z"))

        for j, (lab, unit, (lo, hi)) in enumerate(rng.sample(LABS, rng.randint(3, 5))):
            val = round(rng.uniform(lo, hi), 1)
            add("MimicObservationLabevents", _res(
                "Observation", f"obs-{i:03d}-{j}", "MimicObservationLabevents",
                subject={"reference": f"Patient/{pk}"},
                encounter={"reference": f"Encounter/{enc_id}"},
                code=_cc(lab),
                valueQuantity={"value": val, "unit": unit},
                effectiveDateTime=f"{start}T12:0{j}:00Z"))

    for profile, items in buckets.items():
        path = out_dir / f"{profile}.ndjson"
        path.write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
        print(f"  wrote {len(items):3d} resources -> {path.name}")
    print(f"Synthetic FHIR for {n_patients} patients in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV-FHIR-style data.")
    ap.add_argument("out", nargs="?", default=str(Path(__file__).parent / "synthetic_fhir"))
    ap.add_argument("--n", type=int, default=5, help="number of synthetic patients")
    ap.add_argument("--seed", type=int, default=424242)
    a = ap.parse_args()
    generate(a.n, Path(a.out), a.seed)
