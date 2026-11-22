"""DEPRECATED - do not use.

This script scored the 28-class fine-tuned models against COCO-80 labels, so class ids did not line up
and the results were invalid. Use lab.py instead (it evaluates every model the same way):

    python lab.py train --data datasets/sahayak28/data.yaml --model yolo26n.pt --tag ft_yolo26n --epochs 20
"""
raise SystemExit(__doc__)
