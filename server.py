"""Root entrypoint for the Medical Report Explanation Agent server.
Delegates to backend.server.
"""

import os
import sys

# Ensure backend package and workspace root are in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.server import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  MedExplain AI — Medical Report Explanation Agent")
    print("  Frontend UI:  http://localhost:8000")
    print("  Swagger Docs: http://localhost:8000/api/docs")
    print("  Health Probe: http://localhost:8000/api/health")
    print("=" * 60 + "\n")
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
