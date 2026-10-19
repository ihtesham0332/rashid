"""
Automated Test Suite for Diabetes Risk Prediction API
=====================================================
Uses FastAPI TestClient to test all endpoints end-to-end
using real patient data from cleaned_diabetes_data.csv.

Run with:
    python test_api.py
"""

import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_root():
    print_header("1. Testing Root Endpoint GET /")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "endpoints" in data
    print("  [PASS] Root endpoint returned status 200 with documentation links.")
    print(f"         Message: '{data['message']}'")


def test_health():
    print_header("2. Testing Health Endpoint GET /health")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "ResNet" in data["model_type"] or data["model_type"] == "DiabetesTabularResNet"
    print("  [PASS] Health check verified:")
    print(f"         Status: {data['status']}")
    print(f"         Model Type: {data['model_type']} ({data['architecture']})")
    print(f"         Accuracy: {data['accuracy']} | Macro ROC-AUC: {data['macro_roc_auc']}")
    print(f"         Classes: {data['target_classes']}")


def test_model_info():
    print_header("3. Testing Model Info Endpoint GET /model-info")
    response = client.get("/model-info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "scaler_means" in data
    assert "feature_names" in data
    print(f"  [PASS] Model info returned {len(data['feature_names'])} features and training size: {data['training_data_rows']} records.")


def test_predictions_real_data():
    print_header("4. Testing Single Prediction POST /predict with Real Patient Data")

    # Real data cases directly from cleaned_diabetes_data.csv:
    # 1. Low risk: Row 34 in cleaned data (HbA1c = 5.1% -> Low)
    low_patient = {
        "Age": 42,
        "gender": "Female",
        "Glucose": 105.84,
        "weight": 70.2,
        "height": 1.65,
        "bmi": 25.75,
        "systolic_bp": 110,
        "diastolic_bp": 73,
        "Hba1c": 5.1,
    }

    # 2. Medium risk (Pre-diabetic): Row 114 in cleaned data (HbA1c = 6.0% -> Medium)
    medium_patient = {
        "Age": 55,
        "gender": "Male",
        "Glucose": 118.0,
        "weight": 82.5,
        "height": 1.72,
        "bmi": 27.89,
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "Hba1c": 6.0,
    }

    # 3. High risk (Diabetic): Row 0 in cleaned data (HbA1c = 7.2% -> High)
    high_patient = {
        "Age": 62,
        "gender": "Female",
        "Glucose": 123.3,
        "weight": 47.0,
        "height": 1.52,
        "bmi": 20.24,
        "systolic_bp": 127,
        "diastolic_bp": 74,
        "Hba1c": 7.2,
    }

    test_cases = [
        ("Low Risk Patient", low_patient, "Low"),
        ("Medium Risk Patient", medium_patient, "Medium"),
        ("High Risk Patient", high_patient, "High"),
    ]

    for label, payload, expected_class in test_cases:
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        body = res.json()
        prediction = body["prediction"]
        assert prediction == expected_class, f"Expected '{expected_class}', got '{prediction}'"
        print(f"  [PASS] {label}:")
        print(f"         Prediction: {prediction} (Expected: {expected_class})")
        print(f"         Risk Description: {body['risk_description']}")


def test_batch_prediction():
    print_header("5. Testing Batch Prediction POST /predict/batch")
    batch_payload = {
        "patients": [
            {
                "Age": 42, "gender": "Female", "Glucose": 105.84,
                "weight": 70.2, "height": 1.65, "bmi": 25.75,
                "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1
            },
            {
                "Age": 62, "gender": "Female", "Glucose": 123.3,
                "weight": 47.0, "height": 1.52, "bmi": 20.24,
                "systolic_bp": 127, "diastolic_bp": 74, "Hba1c": 7.2
            }
        ]
    }
    res = client.post("/predict/batch", json=batch_payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    body = res.json()
    assert body["total"] == 2
    assert body["predictions"][0]["prediction"] == "Low"
    assert body["predictions"][1]["prediction"] == "High"
    print(f"  [PASS] Batch of 2 patients processed successfully. Total: {body['total']}.")


def test_data_endpoints():
    print_header("6. Testing Data Endpoints GET /data/stats & GET /data/sample")
    # stats
    res_stats = client.get("/data/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    print(f"  [PASS] /data/stats total rows: {stats['total_rows']}")
    print(f"         Class distribution: {stats['class_distribution']}")

    # sample
    res_sample = client.get("/data/sample?n=3")
    assert res_sample.status_code == 200
    samples = res_sample.json()
    assert len(samples) == 3
    print(f"  [PASS] /data/sample retrieved {len(samples)} random rows.")


def test_validation_errors():
    print_header("7. Testing Input Validation Error Handling (422 Unprocessable Entity)")
    # Test invalid age (> 120)
    bad_age = {
        "Age": 150, "gender": "Male", "Glucose": 100, "weight": 70,
        "height": 1.7, "bmi": 24, "systolic_bp": 120, "diastolic_bp": 80, "Hba1c": 5.5
    }
    res1 = client.post("/predict", json=bad_age)
    assert res1.status_code == 422, f"Expected 422, got {res1.status_code}"
    print("  [PASS] Correctly rejected invalid Age (150) with HTTP 422.")

    # Test invalid gender ("alien")
    bad_gender = {
        "Age": 45, "gender": "Alien", "Glucose": 100, "weight": 70,
        "height": 1.7, "bmi": 24, "systolic_bp": 120, "diastolic_bp": 80, "Hba1c": 5.5
    }
    res2 = client.post("/predict", json=bad_gender)
    assert res2.status_code == 422, f"Expected 422, got {res2.status_code}"
    print("  [PASS] Correctly rejected invalid gender ('Alien') with HTTP 422.")


def test_flexible_inputs():
    print_header("8. Testing Flexible Input Handling (Auto Height cm, Lowercase Keys, Auto BMI)")
    # Test lowercase keys, height in cm (165 cm instead of 1.65 m), and omitted BMI
    flexible_patient = {
        "age": 42,
        "gender": "female",
        "glucose": 105.84,
        "weight": 70.2,
        "height": 165,  # Given in cm! Should auto-convert to 1.65 m
        "systolic_bp": 110,
        "diastolic_bp": 73,
        "hba1c": 5.1,
        # bmi omitted! Should auto-calculate
    }
    res = client.post("/predict", json=flexible_patient)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body["prediction"] == "Low"
    assert round(body["input_data"]["height"], 2) == 1.65
    assert body["input_data"]["bmi"] is not None
    print("  [PASS] Auto-converted height in cm (165 -> 1.65m), auto-calculated BMI, and normalized lowercase keys.")


def test_root_post():
    print_header("9. Testing Root POST Fallback (Prevent 405 Method Not Allowed)")
    # 1. Empty POST to /
    res1 = client.post("/")
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    # 2. Patient POST to /
    patient = {
        "Age": 42, "gender": "Female", "Glucose": 105.84,
        "weight": 70.2, "height": 1.65, "bmi": 25.75,
        "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1
    }
    res2 = client.post("/", json=patient)
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
    assert res2.json()["prediction"] == "Low"
    print("  [PASS] Root POST handled gracefully with 200 OK without 405 error.")


def test_agent_and_mcp_discovery():
    print_header("10. Testing Agent Card & MCP Protocol Discovery (Prevent 404 Probes)")
    # 1. Agent card
    res_agent = client.get("/.well-known/agent.json")
    assert res_agent.status_code == 200, f"Expected 200, got {res_agent.status_code}"
    assert "Diabetes Risk Prediction" in res_agent.json()["name"]

    # 2. OpenAI Plugin manifest
    res_plugin = client.get("/.well-known/ai-plugin.json")
    assert res_plugin.status_code == 200, f"Expected 200, got {res_plugin.status_code}"

    # 3. MCP initialize
    mcp_init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    res_mcp_init = client.post("/mcp", json=mcp_init)
    assert res_mcp_init.status_code == 200, f"Expected 200, got {res_mcp_init.status_code}"
    assert "protocolVersion" in res_mcp_init.json()["result"]

    # 4. MCP tools/list
    mcp_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    res_mcp_tools = client.post("/mcp", json=mcp_tools)
    assert res_mcp_tools.status_code == 200, f"Expected 200, got {res_mcp_tools.status_code}"
    assert len(res_mcp_tools.json()["result"]["tools"]) >= 1

    # 5. MCP tools/call
    mcp_call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "predict_diabetes_risk",
            "arguments": {
                "Age": 42, "gender": "Female", "Glucose": 105.84,
                "weight": 70.2, "height": 1.65, "bmi": 25.75,
                "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1
            }
        }
    }
    res_mcp_call = client.post("/mcp", json=mcp_call)
    assert res_mcp_call.status_code == 200, f"Expected 200, got {res_mcp_call.status_code}"

    print("  [PASS] A2A Agent Card, OpenAI plugin, and MCP JSON-RPC protocol tests all passed (200 OK).")


if __name__ == "__main__":
    print("\nStarting Diabetes Risk Prediction API Automated Test Suite...")
    try:
        test_root()
        test_health()
        test_model_info()
        test_predictions_real_data()
        test_batch_prediction()
        test_data_endpoints()
        test_validation_errors()
        test_flexible_inputs()
        test_root_post()
        test_agent_and_mcp_discovery()
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED SUCCESSFULLY! (10/10 suites passed)")
        print("=" * 60 + "\n")
    except AssertionError as err:
        print(f"\n[FAIL] Test assertion failed: {err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)

