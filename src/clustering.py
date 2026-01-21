import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch, DBSCAN, MeanShift, OPTICS, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from skfuzzy import cmeans


def safe_silhouette_score(X, labels):
    """Безопасный silhouette с проверками"""
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_samples = len(labels)

    if n_labels < 2:
        return -1.0, 0
    if n_labels >= n_samples:
        return -1.0, 0

    try:
        score = silhouette_score(X, labels)
        return float(score), len(labels)
    except ValueError as e:
        print(f"Silhouette error: {e}")
        return -1.0, 0


def matrices_to_priorities_pca(data):
    """
    Преобразует матрицы оценок в векторы глобальных приоритетов экспертов
    с использованием PCA (1-я главная компонента).

    Args:
        data: dict с ключами 'alternatives', 'criteria', 'dms', 'parameters'

    Returns:
        dict в формате {'alternatives': [...], 'dms': [{'id':..., 'scores': [...]}], 'parameters': {...}}
    """
    alternatives = data['alternatives']
    priorities = []

    for dm in data['dms']:
        dm_id = dm['id']
        scores_matrix = np.array(dm['scores'])  # shape: (alternatives, criteria)

        # 1. Стандартизация критериев
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scores_matrix)

        # 2. PCA — первая главная компонента
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled).flatten()

        # 3. Делаем значения положительными
        pc1 = pc1 - pc1.min()

        # 4. Нормализация к сумме 1
        if pc1.sum() != 0:
            pc1 = pc1 / pc1.sum()

        priorities.append({
            'id': dm_id,
            'scores': np.round(pc1, 3).tolist()
        })

    params = {k: v for k, v in data['parameters'].items()}

    return {
        'alternatives': alternatives,
        'dms': priorities,
        'parameters': params
    }


def matrices_to_priorities(data):
    """
    Преобразует матрицы оценок по критериям в векторы глобальных приоритетов экспертов.

    Для каждой альтернативы вычисляет глобальный приоритет как среднее арифметическое
    оценок по всем критериям (равные веса критериев), затем нормализует к сумме 1.

    Args:
        data: dict с ключами 'alternatives', 'criteria', 'dms', 'parameters'

    Returns:
        dict в формате {'alternatives': [...], 'dms': [{'id':..., 'scores': [...]}], 'parameters': {...}}
    """
    alternatives = data['alternatives']
    priorities = []

    for dm in data['dms']:
        dm_id = dm['id']
        scores_matrix = np.array(dm['scores'])

        # Усредняем оценки по критериям для каждой альтернативы
        alt_priorities = np.mean(scores_matrix, axis=1)

        # Нормализуем к сумме 1
        alt_priorities = alt_priorities / np.sum(alt_priorities)

        priorities.append({
            'id': dm_id,
            'scores': np.round(alt_priorities, 3).tolist()
        })

    params = {k: v for k, v in data['parameters'].items()}

    return {
        'alternatives': alternatives,
        'dms': priorities,
        'parameters': params
    }


def find_optimal_k(data, k_min=1, k_max=8, scale=True, random_state=42):
    data = matrices_to_priorities_pca(data)
    X = np.array([dm["scores"] for dm in data["dms"]])
    if scale:
        X = StandardScaler().fit_transform(X)

    n_dms = len(data["dms"])
    if n_dms < 2:
        print(f"Мало DMs ({n_dms}), возвращаем k=1")
        return 1

    inertias = []
    safe_k_max = min(k_max, max(3, n_dms - 1))
    ks = list(range(k_min, safe_k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)

    if len(inertias) < 2:
        print("Недостаточно данных для elbow, возвращаем среднее k")
        return max(1, len(ks) // 2)

    first_diff = np.diff(inertias)
    if len(first_diff) < 2:
        print("Недостаточно данных для second_diff, берём k=2")
        return 2

    second_diff = np.diff(first_diff)

    # базовый выбор
    base_idx = np.argmax(second_diff)
    k_base = ks[base_idx + 1]

    # если соседняя точка даёт ещё заметный перегиб — берём k+1
    if base_idx + 1 < len(second_diff):
        ratio = second_diff[base_idx + 1] / second_diff[base_idx]
        if ratio > 0.3:
            return k_base + 1

    return k_base


def clusterization_result_json(data, n_clusters, cluster_method="ward"):
    """
    Выполняет кластеризацию экспертов и внутрикластерный консенсус,
    возвращает результаты в JSON-формате с ordering кластеров.

    Returns:
        str: JSON с результатами кластеризации + ordering
    """
    # Извлекаем данные
    data = matrices_to_priorities_pca(data)
    alternatives = data["alternatives"]
    dms = data["dms"]
    X = np.array([dm["scores"] for dm in dms])

    # Кластеризация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    method = cluster_method.lower()

    # --- АДАПТИВНЫЕ DBSCAN и OPTICS ---
    def adaptive_dbscan(X_scaled, target_k):
        """DBSCAN: уменьшает eps пока не получит target_k кластеров (2<=k<=n_clusters)"""
        eps_start = 3.0
        eps_step = 0.2
        min_samples = min(1, len(X_scaled) // 2)

        for eps in np.arange(eps_start, 0.1, -eps_step):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            unique_labels = len(set(labels) - {-1})

            if 2 <= unique_labels <= target_k:
                print(f"DBSCAN: eps={eps:.2f}, clusters={unique_labels}")
                return labels, unique_labels, {"eps": float(eps), "min_samples": min_samples}

        # Fallback: первый приемлемый результат
        for eps in np.arange(eps_start, 0.1, -eps_step):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            unique_labels = len(set(labels) - {-1})
            if unique_labels >= 2:
                return labels, unique_labels, {"eps": float(eps), "min_samples": min_samples}
        raise ValueError("DBSCAN не нашёл кластеры")

    def adaptive_optics(X_scaled, target_k, max_tries=20):
        """OPTICS: уменьшает max_eps пока не получит target_k кластеров"""
        min_samples = min(1, len(X_scaled) // 2)
        eps_start = 3.0

        for eps in np.arange(eps_start, 0.1, -0.2):
            optics = OPTICS(min_samples=min_samples, max_eps=eps)
            labels = optics.fit_predict(X_scaled)
            unique_labels = len(set(labels) - {-1})

            if 2 <= unique_labels <= target_k:
                print(f"OPTICS: max_eps={eps:.2f}, clusters={unique_labels}")
                return labels, unique_labels, {"max_eps": float(eps), "min_samples": min_samples}

        # Fallback
        for eps in np.arange(eps_start, 0.1, -0.2):
            optics = OPTICS(min_samples=min_samples, max_eps=eps)
            labels = optics.fit_predict(X_scaled)
            unique_labels = len(set(labels) - {-1})
            if unique_labels >= 2:
                return labels, unique_labels, {"max_eps": float(eps), "min_samples": min_samples}
        raise ValueError("OPTICS не нашёл кластеры")

    # --- выбор модели кластеризации ---
    if method == "ward":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = model.fit_predict(X_scaled)
        membership = np.zeros((X.shape[0], n_clusters))
        hard_labels = model.fit_predict(X_scaled)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        for i, label in enumerate(labels):
            membership[i, label] = 1.0
        params = {}

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        hard_labels = kmeans.fit_predict(X_scaled)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        membership = np.zeros((X.shape[0], n_clusters))
        for i, label in enumerate(labels):
            membership[i, label] = 1.0
        params = {}

    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X_scaled)
        membership = gmm.predict_proba(X_scaled)
        hard_labels = np.argmax(membership, axis=1)  # жёсткие метки для silhouette
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        params = {}

    elif method == "spectral":
        spec = SpectralClustering(
            n_clusters=n_clusters,
            affinity="rbf",
            assign_labels="kmeans",
            random_state=42
        )
        labels = spec.fit_predict(X_scaled)
        hard_labels = spec.fit_predict(X_scaled)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        membership = np.zeros((X.shape[0], n_clusters))
        for i, label in enumerate(labels):
            membership[i, label] = 1.0
        params = {}

    elif method == "birch":
        birch = Birch(n_clusters=n_clusters)
        labels = birch.fit_predict(X_scaled)
        hard_labels = birch.fit_predict(X_scaled)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        membership = np.zeros((X.shape[0], n_clusters))
        for i, label in enumerate(labels):
            if 0 <= label < n_clusters:
                membership[i, label] = 1.0
        params = {}

    elif method == "dbscan":
        labels, k_found, params = adaptive_dbscan(X_scaled, n_clusters)
        unique_labels = sorted(l for l in set(labels) if l != -1)
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        hard_labels = np.array([label_map[l] if l != -1 else -1 for l in labels])  # сохраняем шум как -1

        # Silhouette только для кластеризованных точек (без шума)
        core_mask = hard_labels != -1
        if np.sum(core_mask) > 1:
            silhouette, silhouette_n_points = safe_silhouette_score(X_scaled[core_mask], hard_labels[core_mask])
        else:
            silhouette = -1.0  # если все шум
        n_clusters = k_found
        membership = np.zeros((X.shape[0], n_clusters))
        for i, lab in enumerate(hard_labels):
            if lab >= 0 and lab < n_clusters:  # исключаем шум из membership
                membership[i, lab] = 1.0

    elif method == "optics":
        labels, k_found, params = adaptive_optics(X_scaled, n_clusters)
        unique_labels = sorted(l for l in set(labels) if l != -1)
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        hard_labels = np.array([label_map[l] if l != -1 else -1 for l in labels])
        # Silhouette ТОЛЬКО для кластеризованных точек (без шума)
        core_mask = hard_labels != -1
        if np.sum(core_mask) > 1:  # минимум 2 точки
            silhouette, silhouette_n_points = safe_silhouette_score(X_scaled[core_mask], hard_labels[core_mask])
        else:
            silhouette = -1.0

    elif method == "mean_shift":
        ms = MeanShift()
        labels = ms.fit_predict(X_scaled)
        hard_labels = ms.fit_predict(X_scaled)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        unique_labels = sorted(set(labels))
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        hard_labels = np.array([label_map[l] for l in labels])
        k_found = len(unique_labels)
        membership = np.zeros((X.shape[0], k_found))
        for i, lab in enumerate(hard_labels):
            membership[i, lab] = 1.0
        n_clusters = k_found
        params = {}
    elif method == "fanny":
        # FANNY-like fuzzy clustering с fuzziness=2 (memb.exp=2 по умолчанию в fanny)
        cntr, u_orig, u0, d, jm, p, fpc = cmeans(
            data=np.transpose(X_scaled),  # skfuzzy ожидает features x samples
            c=n_clusters,
            m=2.0,  # fuzziness parameter, аналог memb.exp в fanny
            error=1e-5,
            maxiter=1000,
            init=None,
            seed=42
        )

        # membership = u.T для samples x clusters
        membership = np.fmax(u_orig.T, 1e-10)  # transpose и избежать нулевых
        membership = membership / np.sum(membership, axis=1, keepdims=True)  # нормализовать если нужно

        # Hard labels для silhouette: argmax
        hard_labels = np.argmax(membership, axis=1)
        silhouette, silhouette_n_points = safe_silhouette_score(X_scaled, hard_labels)
        params = {'fuzziness': 2.0, 'fpc': fpc}  # partition coefficient аналогично fanny$coeff


    else:
        raise ValueError(f"Неизвестный метод кластеризации: {cluster_method}")

    # (PCA ordering + консенсус)...
    dms_ids = [dm["id"] for dm in dms]
    hard_labels = np.argmax(membership, axis=1)

    # Центры кластеров для PCA
    cluster_centers = []
    for c in range(n_clusters):
        member_indices = np.where(hard_labels == c)[0]
        if len(member_indices) > 0:
            center = np.mean(X[member_indices], axis=0)
            cluster_centers.append(center)
        else:
            cluster_centers.append(np.zeros(len(alternatives)))

    cluster_centers = np.array(cluster_centers)

    # PCA-проекция
    pca = PCA(n_components=1)
    center_1d = pca.fit_transform(cluster_centers).flatten()

    # Упорядочивание кластеров
    order_indices = np.argsort(center_1d)
    clustering_order = {}
    for order_pos, cluster_id in enumerate(order_indices):
        member_indices = np.where(hard_labels == cluster_id)[0]
        ordered_experts = [dms_ids[i] for i in member_indices]
        clustering_order[str(order_pos)] = ordered_experts  # строка для JSON

    # Функции консенсуса
    def compute_respect_weights(vectors):
        n = vectors.shape[0]
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                weights[i, j] = np.exp(-dist)
        return weights

    def consensus_within_cluster(cluster_vectors, max_iter=1000, epsilon=1e-10):
        P = cluster_vectors.copy()
        for iteration in range(max_iter):
            weights = compute_respect_weights(P)
            P_new = np.zeros_like(P)
            for i in range(len(P)):
                denom = np.sum(weights[i])
                if denom > 0:
                    P_new[i] = np.sum(weights[i][:, None] * P, axis=0) / denom
                else:
                    P_new[i] = P[i]
            diff = np.linalg.norm(P_new - P)
            P = P_new
            if diff < epsilon:
                break
        return P

    # JSON результат
    result = {
        "method": method,
        "n_clusters": int(n_clusters),
        "params": params,
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "silhouette_score": float(silhouette),
        "silhouette_n_points": int(silhouette_n_points),
        "clustering_order": clustering_order,
        "clusters": []
    }

    # Кластеры
    for c in range(n_clusters):
        member_indices = np.where(hard_labels == c)[0]
        cluster_dm_ids = [dms_ids[i] for i in member_indices]
        if len(cluster_dm_ids) > 0:
            cluster_vectors = X[member_indices]
            consensus = consensus_within_cluster(cluster_vectors)
            representative_vector = consensus[0].tolist()

            cluster_info = {
                "cluster_id": int(c),
                "experts": cluster_dm_ids,
                "vector": representative_vector
            }
            result["clusters"].append(cluster_info)
    return result
