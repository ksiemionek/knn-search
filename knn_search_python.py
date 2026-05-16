import math


def python_find_top_k(query, vectors, topk):
    results = []
    for idx, vec in enumerate(vectors):
        dot_prod = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(query, vec):
            dot_prod += x * y
            norm_a += x * x
            norm_b += y * y

        if norm_a == 0.0 or norm_b == 0.0:
            sim = 0.0
        else:
            sim = dot_prod / (math.sqrt(norm_a) * math.sqrt(norm_b))
        results.append((idx, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topk]
