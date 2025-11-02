# Extension 2: Retriever Surgery - Results (FIXED)

## Leakage Metrics by Configuration

| Config | Chunk | K | Retriever | ROUGE-L ↓ | Verbatim 5g ↓ | Verbatim 4g ↓ | Length | N |
|--------|-------|---|-----------|-----------|---------------|---------------|--------|---|
| chunk256_k8_bm25     | 256 | 8 | bm25   | 0.215 | 0.468 | 0.493 | 269.0 | 50 |
| chunk256_k8_dense    | 256 | 8 | dense  | 0.234 | 0.510 | 0.530 | 232.5 | 50 |
| chunk256_k8_hybrid   | 256 | 8 | hybrid | 0.251 | 0.468 | 0.501 | 266.7 | 50 |
| chunk128_k8_hybrid   | 128 | 8 | hybrid | 0.327 | 0.436 | 0.457 | 237.4 | 50 |
| chunk128_k8_bm25     | 128 | 8 | bm25   | 0.368 | 0.401 | 0.430 | 283.4 | 50 |
| chunk128_k8_dense    | 128 | 8 | dense  | 0.372 | 0.400 | 0.439 | 239.5 | 50 |
| chunk64_k8_bm25      |  64 | 8 | bm25   | 0.398 | 0.271 | 0.298 | 238.4 | 50 |
| chunk256_k4_bm25     | 256 | 4 | bm25   | 0.403 | 0.417 | 0.451 | 284.4 | 50 |
| chunk64_k8_dense     |  64 | 8 | dense  | 0.441 | 0.328 | 0.354 | 266.3 | 50 |
| chunk256_k4_hybrid   | 256 | 4 | hybrid | 0.446 | 0.529 | 0.551 | 286.5 | 50 |
| chunk256_k4_dense    | 256 | 4 | dense  | 0.460 | 0.578 | 0.598 | 257.8 | 50 |
| chunk64_k8_hybrid    |  64 | 8 | hybrid | 0.473 | 0.313 | 0.343 | 272.0 | 50 |
| chunk64_k2_bm25      |  64 | 2 | bm25   | 0.479 | 0.207 | 0.238 | 120.7 | 50 |
| chunk128_k4_hybrid   | 128 | 4 | hybrid | 0.490 | 0.408 | 0.432 | 240.0 | 50 |
| chunk64_k2_hybrid    |  64 | 2 | hybrid | 0.528 | 0.278 | 0.316 | 121.4 | 50 |
| chunk64_k2_dense     |  64 | 2 | dense  | 0.536 | 0.305 | 0.327 | 166.1 | 50 |
| chunk128_k4_bm25     | 128 | 4 | bm25   | 0.542 | 0.424 | 0.452 | 282.1 | 50 |
| chunk256_k2_hybrid   | 256 | 2 | hybrid | 0.559 | 0.443 | 0.472 | 252.5 | 50 |
| chunk64_k4_bm25      |  64 | 4 | bm25   | 0.563 | 0.277 | 0.313 | 206.9 | 50 |
| chunk64_k4_dense     |  64 | 4 | dense  | 0.584 | 0.350 | 0.375 | 240.1 | 50 |
| chunk128_k2_bm25     | 128 | 2 | bm25   | 0.608 | 0.401 | 0.429 | 224.9 | 50 |
| chunk64_k4_hybrid    |  64 | 4 | hybrid | 0.617 | 0.365 | 0.394 | 231.8 | 50 |
| chunk256_k2_dense    | 256 | 2 | dense  | 0.624 | 0.508 | 0.536 | 240.8 | 50 |
| chunk256_k2_bm25     | 256 | 2 | bm25   | 0.644 | 0.465 | 0.502 | 279.1 | 50 |
| chunk128_k2_hybrid   | 128 | 2 | hybrid | 0.650 | 0.472 | 0.495 | 166.2 | 50 |
| chunk128_k4_dense    | 128 | 4 | dense  | 0.675 | 0.565 | 0.589 | 258.4 | 50 |
| chunk128_k2_dense    | 128 | 2 | dense  | 0.720 | 0.488 | 0.521 | 175.7 | 50 |

## Key Findings
- **Lowest leakage config**: chunk256_k8_bm25
- **Highest leakage config**: chunk128_k2_dense