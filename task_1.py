# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------
# CONFIGURATION
# -------------------------------
API_KEY = "dea124f675bba71d775f7510ee021910"   # Replace with your OpenWeatherMap API key
CITY = "Mumbai"
COUNTRY_CODE = "IN"

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# -------------------------------
# FETCH WEATHER DATA FROM API
# -------------------------------
def fetch_weather_data(city, country, api_key):
    params = {
        "q": f"{city},{country}",
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        print("✅ Data fetched successfully!")
        return response.json()
    else:
        print("❌ Error fetching data:", response.status_code)
        return None


# -------------------------------
# PROCESS WEATHER DATA
# -------------------------------
def process_weather_data(raw_data):
    weather_list = raw_data["list"]

    processed_data = []

    for entry in weather_list:
        processed_data.append({
            "DateTime": datetime.fromtimestamp(entry["dt"]),
            "Temperature (°C)": entry["main"]["temp"],
            "Feels Like (°C)": entry["main"]["feels_like"],
            "Humidity (%)": entry["main"]["humidity"],
            "Pressure (hPa)": entry["main"]["pressure"],
            "Wind Speed (m/s)": entry["wind"]["speed"],
            "Weather Condition": entry["weather"][0]["main"]
        })

    df = pd.DataFrame(processed_data)
    return df


# -------------------------------
# DATA VISUALIZATION DASHBOARD
# -------------------------------
def create_dashboard(df, city):

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # 1. Temperature Trend
    plt.subplot(2, 2, 1)
    sns.lineplot(
        x="DateTime",
        y="Temperature (°C)",
        data=df,
        marker="o",
        color="red"
    )
    plt.title(f"Temperature Trend in {city}")
    plt.xlabel("Date & Time")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)

    # 2. Humidity Variation
    plt.subplot(2, 2, 2)
    sns.lineplot(
        x="DateTime",
        y="Humidity (%)",
        data=df,
        marker="o",
        color="blue"
    )
    plt.title("Humidity Variation")
    plt.xlabel("Date & Time")
    plt.ylabel("Humidity (%)")
    plt.xticks(rotation=45)

    # 3. Wind Speed Analysis
    plt.subplot(2, 2, 3)
    sns.barplot(
        x=df["DateTime"].dt.date,
        y="Wind Speed (m/s)",
        data=df,
        errorbar=None  # Updated
    )
    plt.title("Average Wind Speed Per Day")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (m/s)")
    plt.xticks(rotation=45)

    # 4. Weather Condition Frequency
    plt.subplot(2, 2, 4)
    weather_counts = df["Weather Condition"].value_counts()

    sns.barplot(
        x=weather_counts.index,
        y=weather_counts.values,
        hue=weather_counts.index,
        palette="viridis",
        legend=False
    )
    plt.title("Weather Condition Frequency")
    plt.xlabel("Condition")
    plt.ylabel("Count")

    plt.suptitle(
        f"Weather Forecast Dashboard for {city}",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    print("🌦️ Weather Data API Integration & Visualization Project")

    raw_data = fetch_weather_data(CITY, COUNTRY_CODE, API_KEY)

    if raw_data:
        df = process_weather_data(raw_data)

        print("\n📊 Sample Processed Data:")
        print(df.head())

        create_dashboard(df, CITY)


# -------------------------------
# RUN PROGRAM
# -------------------------------
if __name__ == "__main__":
    main()
