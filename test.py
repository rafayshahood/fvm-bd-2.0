import requests
from pathlib import Path
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8888/run-verification"
BASE_DIR = Path("test_cases_fvm")
ID_IMAGE = BASE_DIR / "id-card.jpg"
SUMMARY_FILE = "test_results_summary.json"

# Folder-based test expectations
TEST_CASES = {
    "C5 - Anti Spoofing": {
        "folder": BASE_DIR / "C5 - Anti Spoofing",
        "expect_error": "spoof"
    },
    "C4 - Glasses vs No Glasses": {
        "folder": BASE_DIR / "C4 - Glasses vs No Glasses",
        "expect_error": "glasses"
    },
    "All Conditions Met": {
        "folder": BASE_DIR / "All Conditions Met",
        "expect_error": None  # Should pass
    }
}

results_log = []

def run_test(case_name, folder, expect_error):
    print(f"\n=== Running test: {case_name} ===")

    videos = list(folder.glob("*.mp4")) + list(folder.glob("*.mov"))
    if not videos:
        print("⚠️ No videos found in", folder)
        return

    for video in videos:
        print(f"🧪 Testing: {video.name}")
        files = {
            "id_image": open(ID_IMAGE, "rb"),
            "video": open(video, "rb")
        }

        try:
            response = requests.post(API_URL, files=files)
            text = response.text
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                print("❌ Invalid JSON returned from backend.")
                result = {"error": "Invalid JSON", "raw": text}
        except Exception as e:
            print(f"❌ Request failed: {e}")
            result = {"error": str(e)}
        finally:
            files["id_image"].close()
            files["video"].close()

        record = {
            "case": case_name,
            "video": video.name,
            "expected": expect_error or "pass",
        }

        if "error" in result:
            error_msg = result["error"].lower()
            record["status"] = "error"
            record["message"] = error_msg

            if expect_error and expect_error in error_msg:
                print(f"✅ Correctly failed with expected error: {error_msg}")
                record["verdict"] = "pass"
            elif expect_error:
                print(f"❌ Wrong error. Got: {error_msg}")
                record["verdict"] = "fail"
            else:
                print(f"❌ Unexpected failure: {error_msg}")
                record["verdict"] = "fail"
        else:
            record["status"] = "success"
            record["status_text"] = result.get("status")
            record["score"] = result.get("score")
            record["average_score"] = result.get("average_score")

            if expect_error:
                print("❌ Should have failed, but passed.")
                record["verdict"] = "fail"
            else:
                print("✅ Passed.")
                record["verdict"] = "pass"

        results_log.append(record)

def main():
    for name, config in TEST_CASES.items():
        run_test(name, config["folder"], config["expect_error"])

    # Save full test summary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = Path(SUMMARY_FILE)
    with summary_path.open("w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n📁 Test summary saved to: {summary_path.resolve()}")

if __name__ == "__main__":
    main()