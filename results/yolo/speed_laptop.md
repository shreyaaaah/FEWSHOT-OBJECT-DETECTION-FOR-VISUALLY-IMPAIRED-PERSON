# CPU speed measured on the project laptop (`python lab.py speed`, one run, CPU only)

| Model | Params | GFLOPs (fused) | ms/frame | approx. FPS |
|---|---|---|---|---|
| yolo11n | 2.62 M | 6.5 | 118.6 | 8.4 |
| yolo26n | 2.57 M | 5.5 | 108.7 | 9.2 |
| yolo26s | 10.01 M | 20.9 | 283.2 | 3.5 |

Notes: single run on one machine; the ~8% yolo26n vs yolo11n gap is small. On Colab CPU the order was
reversed (94 / 120 / 235 ms), so speed is machine-dependent: quote the laptop numbers for the demo machine.
