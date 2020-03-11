#!/bin/bash

python worker.py -c configs/gen_configs/gen1.yaml

python worker.py -c configs/gen_configs/gen2.yaml

python worker.py -c configs/gen_configs/gen3.yaml

python worker.py -c configs/gen_configs/gen4.yaml

python worker.py -c configs/gen_configs/gen5.yaml

python worker.py -c configs/gen_configs/gen6.yaml



python worker.py -c configs/gen_configs/gen1_seg.yaml

python worker.py -c configs/gen_configs/gen2_seg.yaml

python worker.py -c configs/gen_configs/gen3_seg.yaml

python worker.py -c configs/gen_configs/gen4_seg.yaml

python worker.py -c configs/gen_configs/gen5_seg.yaml

python worker.py -c configs/gen_configs/gen6_seg.yaml



python worker.py -c configs/gen_configs/gen1_og_colors.yaml

python worker.py -c configs/gen_configs/gen2_og_colors.yaml

python worker.py -c configs/gen_configs/gen3_og_colors.yaml

python worker.py -c configs/gen_configs/gen4_og_colors.yaml

python worker.py -c configs/gen_configs/gen5_og_colors.yaml

python worker.py -c configs/gen_configs/gen6_og_colors.yaml