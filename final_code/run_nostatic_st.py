"""Entry point using the shared manuscript-formula implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from papergraph.cli import main

if __name__ == "__main__":
    main(['ablation', '--variants', 'zero', 'graph2'] + sys.argv[1:])
