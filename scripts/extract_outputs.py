import json
from pathlib import Path

for path in sorted(Path("outputs").glob("transformer_*/held_out_*.json")):
    result = json.loads(path.read_text())
    print(
        path,
        f"accuracy={result['test_accuracy']:.4f}",
        f"balanced={result['balanced_accuracy']:.4f}",
        f"majority={result['majority_accuracy']:.4f}",
    )

