import argparse

import requests

GROUND_TRUTH_FILE = "ground_truth.gml"
SAMPLES_FILE = "samples.csv"


def upload_results(dataset_upload_url: str, api_host: str):
    samples_files = {"file": open(SAMPLES_FILE, "rb")}
    response = requests.put(url=dataset_upload_url, files=samples_files)
    response.raise_for_status()
    json = response.json()
    print(json)
    assert "id" in json, f"id was not found in json {json}"
    dataset_id = json["id"]

    ground_truth_upload_url = f"http://{api_host}/api/dataset/{dataset_id}/ground-truth"
    ground_truth_files = {"graph_file": open(GROUND_TRUTH_FILE, "rb")}
    response = requests.post(url=ground_truth_upload_url, files=ground_truth_files)
    print(response.status_code)
    print(response.json())
    response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result dataset uploader')
    parser.add_argument('--uploadEndpoint', type=str, required=True,
                        help='Endpoint to upload the dataset')
    parser.add_argument('--apiHost', type=str, required=True,
                        help='Url of backend')
    args = parser.parse_args()
    upload_results(args.uploadEndpoint, args.apiHost)

