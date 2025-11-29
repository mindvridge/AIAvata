"""MuseTalk 임포트 테스트"""

import sys
from pathlib import Path

# MuseTalk 경로 추가
musetalk_path = Path("external/MuseTalk").resolve()
if musetalk_path.exists():
    sys.path.insert(0, str(musetalk_path))
    print(f"✅ Added to path: {musetalk_path}")
else:
    print(f"❌ Path not found: {musetalk_path}")
    exit(1)

print("\n[1] Testing AudioProcessor import...")
try:
    from musetalk.utils.audio_processor import AudioProcessor
    print("✅ AudioProcessor imported successfully")
except Exception as e:
    print(f"❌ AudioProcessor import failed: {e}")

print("\n[2] Testing UNet import...")
try:
    from musetalk.models.unet import UNet
    print("✅ UNet imported successfully")
except Exception as e:
    print(f"❌ UNet import failed: {e}")

print("\n[3] Testing FaceParsing import...")
try:
    from musetalk.utils.face_parsing import FaceParsing
    print("✅ FaceParsing imported successfully")
except Exception as e:
    print(f"❌ FaceParsing import failed: {e}")

print("\n[4] Testing utilities import...")
try:
    from musetalk.utils.utils import load_all_model
    print("✅ Utilities imported successfully")
except Exception as e:
    print(f"⚠️  Utilities import warning: {e}")

print("\n[5] Checking model files...")
model_dir = Path("models/musetalk/musetalkV15")
if model_dir.exists():
    print(f"✅ Model directory exists: {model_dir}")
    config_file = model_dir / "musetalk.json"
    unet_file = model_dir / "unet.pth"
    if config_file.exists():
        print(f"✅ Config file found: {config_file}")
    if unet_file.exists():
        size_mb = unet_file.stat().st_size / (1024 * 1024)
        print(f"✅ UNet model found: {unet_file} ({size_mb:.2f} MB)")
else:
    print(f"❌ Model directory not found: {model_dir}")

print("\n" + "=" * 60)

