"""Entry point using the shared manuscript-formula implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from papergraph.cli import main

if __name__ == "__main__":
    main(['train', '--variant', 'gcn_lstm'] + sys.argv[1:])
