import requests

# Define the input data
data = {
    "num_passengers": 5,
    "channel": "Internet",
    "trip": "RoundTrip",
    "purchase_lead": 262,
    "length_of_stay": 7,
    "flight_hour": 14,
    "day": "Sat",
    "route": "CGKICN",
    "booking_origin": "India",
    "wants_extra_baggage": 1,
    "wants_preferred_seat": 0,
    "wants_in_flight_meals": 1,
    "flight_duration": 3.5
}

# Send the POST request
response = requests.post("http://0.0.0.0:8000/predict", json=data)

# Check the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)
