import requests
import csv

def download_mock_data():
    # Fetch data from March 18 to March 20 2026
    url = "https://api.carbonintensity.org.uk/intensity/2026-03-18T00:00Z/2026-03-20T23:59Z"
    
    print("Fetching historical data from UK Grid...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()['data']
        
        with open("mock_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["from", "to", "intensity_forecast", "intensity_actual", "index"])
            
            for entry in data:
                writer.writerow([
                    entry['from'],
                    entry['to'],
                    entry['intensity']['forecast'],
                    entry['intensity']['actual'],
                    entry['intensity']['index']
                ])
        print("Success! mock_data.csv has been created.")
    else:
        print(f"Failed to fetch data: {response.status_code}")

download_mock_data()