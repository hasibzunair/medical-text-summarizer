import requests
import glob

# Set the URL
API_URL = "http://0.0.0.0:8000"


def run_health():
    """Run /health endpoint."""

    response = requests.get(f"{API_URL}/health")
    print("Health endpoint reply:")
    print(response.json())
    print("\n")


def run_summarize():
    """Run /summarize endpoint."""

    print("Reading provided notes. \n")
    txt_files = glob.glob("./datasets/notes/**/*.txt", recursive=True)
    if not txt_files:
        print("No text files found.")
    else:
        for file_path in txt_files:
            with open(file_path, "r") as f:
                text_content = f.read()

            # create payload with text and optionally a role
            payload = {
                "text": text_content,
                "clinical_role": "physician",  # could be nurse, radiologist, neurologist
            }

            # send payload
            response = requests.post(f"{API_URL}/summarize", json=payload)

            if response.status_code == 200:
                result = response.json()
                print(f"Summary for {file_path}:")
                print("\n")
                print(result["summary"])
                print("\n")
                print(
                    f"Tokens used: {result['tokens']} | Processing time: {result['processing_time']:.2f}s"
                )
                print("\n")
                print("List of References:")
                for link in result["references"]:
                    print(link)
                    print("\n")
            else:
                print(
                    f"Could not summarize {file_path}. Status code: {response.status_code}, Details: {response.text}"
                )


def run_summarize_non_medical():
    """Run /summarize endpoint with non medical text."""

    text = "I like outdoor activities such as hiking, playing badmintor and football. Nature walk is also something is enjoy very much."
    payload = {"text": text, "clinical_role": "physician"}
    response = requests.post(f"{API_URL}/summarize", json=payload)
    if response.status_code == 200:
        result = response.json()
        print("Summary:")
        print("Expected: <|no_medical_text|>")
        print("Actual:  ", result["summary"])
        print("\n")
    else:
        print(
            f"Failed to summarize non-medical text. Status code: {response.status_code}, Details: {response.text}"
        )


def run_feedback():
    """Run /feedback endpoint."""

    payload = {"feedback": "The summary was good."}
    response = requests.post(f"{API_URL}/feedback", json=payload)
    print("Feedback endpoint reply:")
    print(response.json())


if __name__ == "__main__":
    run_health()
    run_summarize()
    run_summarize_non_medical()
    run_feedback()
