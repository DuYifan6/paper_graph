from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

FEATURES = ('M_RSI_6_5', 'D_pct_chg', 'M_MACD_1', 'D_mainforce_strength',
            'D_bull_bear_ratio', 'D_intraday_return', 'M_high_1',
            'D_overnight_return', 'M_MACD_hist_1')
SCHEMA = 'papergraph-v2-undirected-union-formula-loss'

@dataclass(frozen=True)
class Config:
    lookback: int = 20
    window: int = 20
    neighbors: int = 40
    threshold: float = .10
    hidden: int = 128
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    laplacian: float = .001
    min_nodes: int = 50
    fit_end: str = '2024-06-05'
    validation_start: str = '2024-06-06'
    validation_end: str = '2024-12-31'
    test_start: str = '2025-01-02'
    test_end: str = '2025-11-17'
    expected_test_days: int = 211
    protocol: str = 'full_day'
    def __post_init__(self):
        if min(self.lookback,self.window,self.neighbors,self.hidden,self.epochs,self.patience,self.min_nodes)<1:
            raise ValueError('Positive dimensions and training limits required')
        if not 0 < self.threshold <= 1 or self.laplacian < 0:
            raise ValueError('Invalid graph threshold or penalty')
        if self.protocol not in ('full_day','hybrid1455'):
            raise ValueError('Unknown information protocol')
    def dictionary(self): return asdict(self)
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.dictionary(),sort_keys=True).encode()).hexdigest()[:16]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
