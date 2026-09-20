.venv/Scripts/activate.ps1
rm logs/*
rm data/results/*
rm memory.db

python scripts/local_eval.py --data .\data\prepared\personamem-v2-32k.jsonl --convs 1 --limit 30  --output data/results/personamem-v2-32k.jsonl

# python scripts/local_eval.py --data .\data\prepared\personamem-v2-32k.jsonl --convs 2 --limit 50  --output data/results/personamem-v2-32k.jsonl
