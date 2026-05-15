use pyo3::prelude::*;

/// kNN search module
#[pymodule]
mod knn_search {
  use pyo3::prelude::*;
  use rayon::prelude::*;

  // Calculates cosine similarity between two vectors
  // https://en.wikipedia.org/wiki/Cosine_similarity
  fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot_prod = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
      dot_prod += x * y;
      norm_a += x * x;
      norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
      return 0.0;
    }

    dot_prod / (norm_a.sqrt() * norm_b.sqrt())
  }

  /// Finds top-k most similar vectors to query
  #[pyfunction]
  fn find_top_k(
    query: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    topk: usize,
  ) -> PyResult<Vec<(usize, f64)>> {
    let mut results: Vec<(usize, f64)> = vectors
      .par_iter()
      .enumerate()
      .map(|(idx, vec)| (idx, cosine_similarity(&query, vec)))
      .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(topk);

    Ok(results)
  }
}
