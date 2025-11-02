# Extension 2: Retriever Surgery - Results (FIXED)

## Leakage Metrics by Configuration

| Config | Chunk | K | Retriever | ROUGE-L ↓ | Verbatim 5g ↓ | Verbatim 4g ↓ | Length | N |
|--------|-------|---|-----------|-----------|---------------|---------------|--------|---|
| chunk256_k8_bm25     | 256 | 8 | bm25   | 0.234 | 0.366 | 0.404 | 274.3 | 50 |
| chunk256_k8_hybrid   | 256 | 8 | hybrid | 0.248 | 0.419 | 0.450 | 269.8 | 50 |
| chunk256_k8_dense    | 256 | 8 | dense  | 0.256 | 0.409 | 0.438 | 236.0 | 50 |
| chunk128_k8_bm25     | 128 | 8 | bm25   | 0.344 | 0.373 | 0.403 | 259.1 | 50 |
| chunk128_k8_hybrid   | 128 | 8 | hybrid | 0.344 | 0.286 | 0.314 | 271.0 | 50 |
| chunk128_k8_dense    | 128 | 8 | dense  | 0.385 | 0.390 | 0.407 | 258.8 | 50 |
| chunk256_k4_bm25     | 256 | 4 | bm25   | 0.413 | 0.475 | 0.505 | 268.6 | 50 |
| chunk256_k4_dense    | 256 | 4 | dense  | 0.456 | 0.549 | 0.570 | 242.7 | 50 |
| chunk256_k4_hybrid   | 256 | 4 | hybrid | 0.473 | 0.541 | 0.573 | 268.2 | 50 |
| chunk64_k8_hybrid    |  64 | 8 | hybrid | 0.527 | 0.355 | 0.388 | 271.3 | 50 |
| chunk64_k8_bm25      |  64 | 8 | bm25   | 0.548 | 0.387 | 0.422 | 276.7 | 50 |
| chunk128_k4_bm25     | 128 | 4 | bm25   | 0.557 | 0.443 | 0.474 | 238.9 | 50 |
| chunk256_k2_hybrid   | 256 | 2 | hybrid | 0.594 | 0.510 | 0.538 | 214.3 | 50 |
| chunk64_k4_hybrid    |  64 | 4 | hybrid | 0.601 | 0.434 | 0.462 | 159.1 | 50 |
| chunk128_k4_hybrid   | 128 | 4 | hybrid | 0.607 | 0.470 | 0.497 | 249.9 | 50 |
| chunk64_k2_bm25      |  64 | 2 | bm25   | 0.621 | 0.359 | 0.394 | 105.4 | 50 |
| chunk64_k4_bm25      |  64 | 4 | bm25   | 0.625 | 0.425 | 0.457 | 169.0 | 50 |
| chunk128_k2_bm25     | 128 | 2 | bm25   | 0.631 | 0.417 | 0.447 | 175.1 | 50 |
| chunk256_k2_bm25     | 256 | 2 | bm25   | 0.638 | 0.535 | 0.569 | 255.1 | 50 |
| chunk64_k8_dense     |  64 | 8 | dense  | 0.644 | 0.534 | 0.560 | 260.7 | 50 |
| chunk128_k4_dense    | 128 | 4 | dense  | 0.644 | 0.538 | 0.557 | 239.4 | 50 |
| chunk256_k2_dense    | 256 | 2 | dense  | 0.658 | 0.483 | 0.505 | 221.4 | 50 |
| chunk128_k2_hybrid   | 128 | 2 | hybrid | 0.693 | 0.540 | 0.562 | 171.8 | 50 |
| chunk64_k2_hybrid    |  64 | 2 | hybrid | 0.710 | 0.450 | 0.480 | 92.2 | 50 |
| chunk64_k2_dense     |  64 | 2 | dense  | 0.725 | 0.568 | 0.590 | 100.6 | 50 |
| chunk128_k2_dense    | 128 | 2 | dense  | 0.765 | 0.637 | 0.653 | 153.8 | 50 |
| chunk64_k4_dense     |  64 | 4 | dense  | 0.770 | 0.617 | 0.640 | 185.4 | 50 |

## Key Findings
- **Lowest leakage config**: chunk256_k8_bm25
- **Highest leakage config**: chunk64_k4_dense