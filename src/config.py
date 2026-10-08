from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET = "Late_delivery_risk"

# Gerçek teslim sonucunu sızdıran / hedefle neredeyse aynı bilgi veren kolonlar
LEAKY_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",  # target — features'tan ayrı tutulur
]
