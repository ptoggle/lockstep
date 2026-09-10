# Re-baseline against a clean stock (2026-09-09)

Single-stream, one H100 SXM, both stacks in full CUDA graphs without inductor, REP=3 medians; lockstep over stock
(above 1.0 is faster than stock). `single_<model>.json` holds both arms; `single_<model>_stock2.json` is the stock arm
repeated after the lockstep arm (Qwen2.5-7B). Hosts: lc-handover (7B, Llama, Qwen3, Phi, Mistral, 1M) at load 11-29 on
208 cores; lc-sweep (the Qwen2.5 size sweep) at load 11-17.

| model | TTFT p128 x1 | TTFT p128 x8 | TTFT p2048 x1 | TTFT p2048 x8 | TTFT p8192 x1 | TTFT p8192 x8 | TTFT p8192 x4 | TTFT p32640 x1 | prefill_p128 x32 | decode x1 | decode x8 | decode x32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | 0.785 | 0.792 | 0.846 | 0.884 | 0.918 | 0.871 | 0.878 | 0.864 | 0.872 | 0.919 | 0.903 | 0.861 |
| mistral7b | 0.766 | 0.789 | 0.849 | 0.897 | 0.911 | 0.885 | 0.879 | 0.853 | 0.868 | 0.839 | 0.868 | 0.809 |
| phi4_mini | 0.783 | 0.775 | 0.783 | 0.792 | 0.808 | 0.782 | 0.782 | 0.795 | 0.802 | 0.924 | 0.951 | 0.900 |
| qwen25_14b | 0.823 | 0.865 | 0.934 | 0.954 | 0.956 | 0.938 | 0.936 | 0.893 | 0.979 | 1.077 | 1.074 | 0.989 |
| qwen25_1p5b | 0.600 | 0.595 | 0.491 | 0.490 | 0.531 | 0.508 | 0.515 | 0.655 | 0.525 | 0.506 | 0.503 | 0.485 |
| qwen25_3b | 0.592 | 0.574 | 0.592 | 0.622 | 0.669 | 0.620 | 0.632 | 0.685 | 0.621 | 0.614 | 0.612 | 0.581 |
| qwen25_7b | 0.808 | 0.733 | 0.862 | 0.877 | 0.898 | 0.866 | 0.875 | 0.869 | 0.892 | 0.939 | 0.937 | 0.857 |
| qwen3_8b | 0.559 | 0.425 | 0.447 | 0.468 | 0.497 | 0.487 | 0.489 | 0.554 | 0.459 | 0.734 | 0.716 | 0.715 |

Saturated serving (64 concurrent, `t14_bench.py`, output tokens per second, lockstep over stock):

| model | W1 ShareGPT | W2 decode-heavy | W3 prefill-heavy |
|---|---|---|---|
| qwen25_7b | 0.748 (4033 / 5393) | 0.806 (6579 / 8165) | 0.770 (448 / 582) |
| llama31_8b | 0.725 (3641 / 5025) | 0.776 (5733 / 7390) | 0.794 (413 / 521) |

Long context, Qwen2.5-7B-Instruct-1M (dense), `t16f_long.py`, its own manifest (`rb_qwen1m`, key offset calibrated by prepare):

| row | stock | lockstep | ratio |
|---|---|---|---|
| ttft_p32768_b1 (ttft_ms) | 1096.86 | 1270.97 | 0.863 |
| ttft_p65536_b1 (ttft_ms) | 2949.47 | 3479.51 | 0.848 |
| ttft_p131072_b1 (ttft_ms) | 8794.52 | 10697.86 | 0.822 |
| ttft_p262144_b1 (ttft_ms) | 29406.25 | 36232.87 | 0.812 |
| decode_p131072_b1 (tok_s) | 96.73 | 110.78 | 1.145 |
