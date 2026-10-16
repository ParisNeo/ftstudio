import json
import sys
import time
from pathlib import Path


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    print("noop start", flush=True)
    time.sleep(float(p.get("sleep", 0)))
    if p.get("fail"):
        sys.exit(3)
    (run / "result.json").write_text(json.dumps({"echo": p}))
    print("noop done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
