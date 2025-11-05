import os
import sys

SAM2_ROOT = os.path.dirname(__file__)
SAM2_SUBDIR = os.path.join(SAM2_ROOT, "sam2")

if os.path.exists(SAM2_SUBDIR) and SAM2_SUBDIR not in sys.path:
    sys.path.insert(0, SAM2_SUBDIR)

print("[DEBUG] sam2/__init__.py loaded.")
print("[DEBUG] Added to sys.path:", SAM2_SUBDIR in sys.path)
