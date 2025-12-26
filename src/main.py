from analyze_bias import analyze_bias_aggregation, save_bias_analysis
from clustering import find_optimal_k, clusterization_result_json
from utils import save_json_with_date, load_config


def main(config_path: str):
    """Основной пайплайн кластеризации + анализ bias."""
    # Загрузка данных
    data = load_config(f'config/{config_path}')

    # Определяем оптимальное k
    n_dms = len(data['dms'])
    safe_k_max = min(8, n_dms - 1)
    safe_k_max = max(2, safe_k_max)
    k = find_optimal_k(data, k_min=1, k_max=safe_k_max)
    print(f"Оптимальное k: {k}")

    # Все методы кластеризации
    methods = ['ward', 'kmeans', 'gmm', 'spectral', 'birch', 'dbscan', 'mean_shift']
    final_result = []

    for method in methods:
        print(f"Запуск {method}...")
        result = clusterization_result_json(data, n_clusters=k, cluster_method=method)
        final_result.append(result)

    # 1️⃣ ОСНОВНОЙ РЕЗУЛЬТАТ
    output_file = save_json_with_date(final_result, f"{config_path.replace('.json', '')}_result_clusters")
    print(f"Результаты сохранены: {output_file}")

    # 2️⃣ АНАЛИЗ ИЗОЛЯЦИИ
    print("\n🔍 Анализ предвзятости экспертов...")
    bias_results = analyze_bias_aggregation(final_result)

    bias_filename = f"{config_path.replace('.json', '')}_bias_analysis"
    bias_file = save_bias_analysis(bias_results, bias_filename)
    print(f"💾 Анализ bias: {bias_file}")

    # 3️⃣ ТОП-5 в консоль
    print("\n🏆 ТОП-5 ИЗОЛИРОВАННЫХ ЭКСПЕРТОВ:")
    print("-" * 40)
    for expert, count in list(bias_results.items())[:5]:
        print(f"{expert}: {count} методов")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "data1_from_article.json"
    main(config_path)
