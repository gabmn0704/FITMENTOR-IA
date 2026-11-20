"""Mide /analyze contra el conjunto sintético etiquetado y barre umbrales.

    python -m evals.pose_eval              # informe con los umbrales actuales
    python -m evals.pose_eval --sweep      # barrido de umbrales con evidencia
    python -m evals.pose_eval --json       # métricas en JSON

Para cada ejercicio y dimensión (profundidad: poco profunda/demasiado; alineación) se calcula
precisión y exhaustividad (recall) de la corrección esperada y la tasa de falsos positivos sobre
las poses buenas (cualquier corrección sobre una pose "ok" cuenta como falso positivo).
"""
import argparse
import copy
import json
import sys
from collections import defaultdict

from app import main as pose
from evals.pose_synth import REFERENCE, Sample, all_samples

# Texto de la corrección principal -> tipo detectado.
KIND_MARKERS = (
    ("shallow", ("Baja un poco más", "Baja más el pecho", "Lleva la cadera atrás")),
    ("deep", ("Reduce la profundidad", "Reduce la inclinación")),
    ("bad", ("línea recta",)),
)
# Dimensión de la muestra en la que cae cada tipo de corrección.
KIND_DIMENSION = {"shallow": None, "deep": None, "bad": "line"}


def classify(feedback: list[str], exercise: str) -> dict[str, str]:
    """dimensión -> clase predicha, a partir de TODAS las correcciones devueltas (no solo la primera)."""
    angle_dim = next((dim for dim in REFERENCE[exercise] if dim != "line"), None)
    predicted: dict[str, str] = {}
    if angle_dim:
        predicted[angle_dim] = "ok"
    if "line" in REFERENCE[exercise]:
        predicted["line"] = "ok"
    for text in feedback:
        for kind, markers in KIND_MARKERS:
            if any(marker in text for marker in markers):
                dimension = "line" if kind == "bad" else angle_dim
                if dimension:
                    predicted[dimension] = kind
    return predicted


def run(samples: list[Sample]) -> dict:
    """Matriz de confusión por ejercicio y dimensión."""
    confusion: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    not_analyzed = 0
    for sample in samples:
        response = pose.analyze(
            pose.PoseRequest(exercise=sample.exercise, points=sample.points, frame_width=sample.frame_width, frame_height=sample.frame_height)
        )
        if not response.analyzed:
            not_analyzed += 1
            continue
        predicted = classify(response.feedback if response.feedback != [pose.GOOD_FORM] else [], sample.exercise)
        for dimension, truth in sample.labels.items():
            confusion[(sample.exercise, dimension)][(truth, predicted[dimension])] += 1
    return {"confusion": confusion, "not_analyzed": not_analyzed, "total": len(samples)}


def metrics(result: dict) -> dict:
    out: dict[str, dict] = {}
    for (exercise, dimension), cells in sorted(result["confusion"].items()):
        classes = sorted({truth for truth, _ in cells} | {pred for _, pred in cells})
        per_class = {}
        for cls in classes:
            tp = cells.get((cls, cls), 0)
            fp = sum(count for (truth, pred), count in cells.items() if pred == cls and truth != cls)
            fn = sum(count for (truth, pred), count in cells.items() if truth == cls and pred != cls)
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            per_class[cls] = {"precision": round(precision, 3), "recall": round(recall, 3), "support": tp + fn}
        good_total = sum(count for (truth, _), count in cells.items() if truth == "ok")
        good_flagged = sum(count for (truth, pred), count in cells.items() if truth == "ok" and pred != "ok")
        bad_total = sum(count for (truth, _), count in cells.items() if truth != "ok")
        bad_missed = sum(count for (truth, pred), count in cells.items() if truth != "ok" and pred == "ok")
        out[f"{exercise}.{dimension}"] = {
            "classes": per_class,
            "false_positive_rate": round(good_flagged / good_total, 3) if good_total else 0.0,
            "miss_rate": round(bad_missed / bad_total, 3) if bad_total else 0.0,
        }
    return out


def macro_f1(report: dict) -> float:
    scores = []
    for entry in report.values():
        for cls, values in entry["classes"].items():
            p, r = values["precision"], values["recall"]
            scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def print_report(report: dict, total: int, not_analyzed: int) -> None:
    print(f"Muestras: {total}; no analizadas: {not_analyzed}")
    print(f"{'ejercicio.dimensión':<18} {'clase':<8} {'precisión':>9} {'recall':>7} {'n':>5}   FP(buenas)  omisión(malas)")
    for key, entry in report.items():
        first = True
        for cls, values in entry["classes"].items():
            tail = f"   {entry['false_positive_rate']:>9.1%}  {entry['miss_rate']:>13.1%}" if first else ""
            print(f"{key if first else '':<18} {cls:<8} {values['precision']:>9.3f} {values['recall']:>7.3f} {values['support']:>5}{tail}")
            first = False
    print(f"F1 macro: {macro_f1(report):.3f}")


def sweep(samples: list[Sample]) -> None:
    """Barre cada umbral manteniendo los demás y muestra F1 macro, FP y omisión de su dimensión."""
    original = copy.deepcopy(pose.LIMITS)
    grids = {
        ("squat", "high"): range(105, 131, 5),
        ("squat", "low"): range(45, 76, 5),
        ("lunge", "high"): range(95, 116, 5),
        ("lunge", "low"): range(55, 86, 5),
        ("push_up", "high"): range(100, 131, 5),
        ("push_up", "low"): range(45, 76, 5),
        ("push_up", "line"): [0.06, 0.08, 0.10, 0.12],
        ("plank", "line"): [0.05, 0.06, 0.08, 0.10, 0.12],
        ("deadlift", "high"): range(135, 166, 5),
        ("deadlift", "low"): range(25, 56, 5),
    }
    try:
        for (exercise, key), values in grids.items():
            print(f"\n{exercise}.{key} (actual: {original[exercise][key]})")
            relevant = [s for s in samples if s.exercise == exercise]
            for value in values:
                pose.LIMITS[exercise][key] = value
                report = metrics(run(relevant))
                fp = max(entry["false_positive_rate"] for entry in report.values())
                miss = max(entry["miss_rate"] for entry in report.values())
                print(f"  {value:>6}: F1 {macro_f1(report):.3f}  FP {fp:.1%}  omisión {miss:.1%}")
            pose.LIMITS[exercise][key] = original[exercise][key]
    finally:
        pose.LIMITS.clear()
        pose.LIMITS.update(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    samples = all_samples(args.per_class, args.seed)
    if args.sweep:
        sweep(samples)
        return 0
    result = run(samples)
    report = metrics(result)
    if args.json:
        print(json.dumps({"macro_f1": round(macro_f1(report), 3), "not_analyzed": result["not_analyzed"], "report": report}, ensure_ascii=False, indent=2))
    else:
        print_report(report, result["total"], result["not_analyzed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
