import urllib.request
import os

URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_dataset():
    print(f"Downloading dataset from: {URL}")
    try:
        urllib.request.urlretrieve(URL, OUTPUT_FILE)
        if os.path.exists(OUTPUT_FILE):
            size_kb = os.path.getsize(OUTPUT_FILE) / 1024
            print(f"Success! Dataset downloaded and saved to '{OUTPUT_FILE}' ({size_kb:.2f} KB).")
        else:
            print("Error: File was not created.")
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")

if __name__ == "__main__":
    download_dataset()
