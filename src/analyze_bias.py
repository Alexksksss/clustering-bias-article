import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def analyze_bias_aggregation(json_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Считает изоляцию с ПРАВИЛЬНЫМ total_experts."""
    all_experts = set()
    for method_data in json_data:
        for cluster in method_data['clusters']:
            all_experts.update(cluster['experts'])

    total_experts = len(all_experts)
    half_experts = total_experts / 2
    print(f"Всего экспертов: {total_experts}, порог: <{half_experts}")

    expert_isolation_count = defaultdict(int)
    for method_data in json_data:
        for cluster in method_data['clusters']:
            if len(cluster['experts']) < half_experts:
                for expert in cluster['experts']:
                    expert_isolation_count[expert] += 1

    for expert in all_experts:
        if expert not in expert_isolation_count:
            expert_isolation_count[expert] = 0

    return dict(sorted(expert_isolation_count.items(), key=lambda x: x[1], reverse=True))


def save_bias_analysis(results: Dict[str, int], base_filename: str) -> str:
    """Сохраняет анализ bias в results/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path("results-COLAB")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = results_dir / f"{base_filename}_{timestamp}.json"

    total_methods = max(results.values()) if results else 0

    min_isolation = min(results.values()) if results else 0
    normal_experts = {exp: score for exp, score in results.items() if score == min_isolation}

    analysis_data = {
        "bias_aggregation": results,

        # ТОП-3 по абсолютному количеству
        "top3_isolated_absolute": dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]),
        #  КРИТИЧЕСКИЕ (>65% методов)
        "critical_bias>65%": {exp: score for exp, score in results.items() if score >= total_methods * 0.65},

        "normal_experts": normal_experts,
        "min_isolation_level": min_isolation,

        "stats": {
            "total_methods": total_methods,
            "total_experts": len(results),
            "avg_isolation": round(np.mean(list(results.values())), 2) if results else 0,
            "min_isolation": min_isolation,
            "max_isolation": max(results.values()) if results else 0,
            "bias_prevalence": round(len([s for s in results.values() if s > min_isolation]) / len(results) * 100, 1)
        },
        "timestamp": timestamp
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=4, ensure_ascii=False)

    return str(filename)
