"""
Application Entrypoint for Hugging Face Spaces & Local Deployment
================================================================
Launches the Diabetes Risk Prediction FastAPI service on the port specified
by the PORT environment variable (default 7860 for Hugging Face Spaces).
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 65)
    print("  Diabetes Risk Prediction API (Deep Tabular ResNet 98.6%)")
    print(f"  Listening on http://0.0.0.0:{port}")
    print("  Interactive Swagger Docs: /docs")
    print("=" * 65)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
