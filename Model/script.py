import sys
import requests

# Set your API key and target output file name
API_KEY = "a78f1d76cf0d66b82c65441c1838791945ad3ff595c311aa"
OUTPUT_FILE = "urlhaus_recent.csv"

# Request endpoint
URL = "https://urlhaus-api.abuse.ch/v2/files/exports/recent.csv"

# Headers including authentication
headers = {
    "Auth-Key": API_KEY,
    "User-Agent": "URLhaus-Downloader/1.0"
}

def download_urlhaus_data(url, output_path, auth_key):
    print(f"Downloading data from URLhaus...")
    
    try:
        # Request with Auth-Key header and streaming enabled
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        # Raise an exception if HTTP request returns a error status code (e.g. 401 Unauthorized, 404)
        response.raise_for_status()
        
        # Write chunks to file to handle memory efficiently
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Download complete! Data saved successfully to: '{output_path}'")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} (Check your API Key / Auth-Key header)")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    if API_KEY == "YOUR-AUTH-KEY-HERE":
        print("Warning: Replace 'YOUR-AUTH-KEY-HERE' with your actual URLhaus API key.")
    
    download_urlhaus_data(URL, OUTPUT_FILE, API_KEY)