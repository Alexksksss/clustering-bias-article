import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent


def convert_paths(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(item) for item in obj]
    return obj


# Безопасная сериализация
def convert_paths(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(item) for item in obj]
    return obj


def save_json_with_date(data: Any, base_filename: str = "data") -> str:
    """Сохраняет JSON с датой в папке results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = PROJECT_ROOT / "results-COLAB"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = results_dir / f"{base_filename}_{timestamp}.json"

    safe_data = convert_paths(data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(safe_data, f, indent=4, ensure_ascii=False)

    print(f"💾 СОХРАНЕНО: {filename}")
    return str(filename)


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из JSON файла."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
