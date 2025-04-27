import express from 'express';
import fetch from 'node-fetch';
import { createObjectCsvWriter } from 'csv-writer';
import cors from 'cors';
import fs from 'fs';
import { DateTime } from 'luxon';
import pkg from 'pg';
import { WebSocketServer } from 'ws';
const { Client } = pkg;

process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = 0;

const wss = new WebSocketServer({ port: 8081 });

const app = express();
const port = 3001;

app.use(express.json());
app.use(cors());
app.use(express.static('public'));

const WINDY_API_KEY = "xp25qby8srkzbQEnidFKMrswgjy3RPOM";

const weatherParameters = [
  "temp", "dewpoint", "wind", "windGust", "precip", "pressure", "rh",
  "lclouds", "mclouds", "hclouds"
];

function getCsvWriter() {
  const filePath = 'weather_data.csv';
  const fileExists = fs.existsSync(filePath);
  const isFileEmpty = fileExists && fs.statSync(filePath).size === 0;
  return createObjectCsvWriter({
    path: filePath,
    header: [
      { id: 'timestamp', title: 'Timestamp' },
      { id: 'place_name', title: 'Place Name' },
      { id: 'latitude', title: 'Latitude' },
      { id: 'longitude', title: 'Longitude' },
      { id: 'temp', title: 'Temperature (°C)' },
      { id: 'dewpoint', title: 'Dew Point (°C)' },
      { id: 'wind', title: 'Wind Speed (km/h)' },
      { id: 'windGust', title: 'Wind Gust (km/h)' },
      { id: 'precip', title: 'Precipitation (mm)' },
      { id: 'pressure', title: 'Pressure (hPa)' },
      { id: 'humidity', title: 'Humidity (%)' },
      { id: 'cloudCover', title: 'Cloud Cover (%)' }
    ],
    append: fileExists && !isFileEmpty
  });
}

const pgClient = new Client({
  host: '127.0.0.1',
  port: 5432,
  user: 'postgres',
  password: '1234',
  database: 'postgres'
});

pgClient.connect()
  .then(() => console.log('Connected to PostgreSQL'))
  .catch(err => console.error('Postgres connection error:', err));

// ADD fetchAndStore function here (MINIMAL change)
async function fetchAndStore(lat, lon, placeName) {
  const requestBody = {
    lat: lat,
    lon: lon,
    model: "gfs",
    parameters: weatherParameters,
    levels: ["surface"],
    key: WINDY_API_KEY
  };

  const response = await fetch('https://api.windy.com/api/point-forecast/v2', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody)
  });

  const data = await response.json();

  if (data.error) {
    throw new Error(data.error);
  }

  const isoTimestamp = DateTime.now().setZone('Asia/Kolkata').toISO();

  const cloudCover = (
    (data["lclouds-surface"]?.[0] || 0) +
    (data["mclouds-surface"]?.[0] || 0) +
    (data["hclouds-surface"]?.[0] || 0)
  ) / 3;

  const weatherData = {
    timestamp: isoTimestamp,
    place_name: placeName,
    latitude: lat,
    longitude: lon,
    temp: data["temp-surface"] ? data["temp-surface"][0] - 273.15 : null,
    dewpoint: data["dewpoint-surface"] ? data["dewpoint-surface"][0] - 273.15 : null,
    wind: data["wind-surface"] ? data["wind-surface"][0] : null,
    windGust: data["windGust-surface"] ? data["windGust-surface"][0] : null,
    precip: data["precip-surface"] ? data["precip-surface"][0] : null,
    pressure: data["pressure-surface"] ? data["pressure-surface"][0] : null,
    humidity: data["rh-surface"] ? data["rh-surface"][0] : null,
    cloudCover: cloudCover
  };

  const columns = Object.keys(weatherData).map(key => `"${key}"`).join(', ');
  const placeholders = Object.values(weatherData).map((_, index) => `$${index + 1}`).join(', ');
  const query = `INSERT INTO weather (${columns}) VALUES (${placeholders})`;
  await pgClient.query(query, Object.values(weatherData));

  return weatherData;
}

app.post('/get-weather', async (req, res) => {
  const { lat, lon, placeName } = req.body;

  try {
    const weatherData = await fetchAndStore(lat, lon, placeName);
    console.log('Windy API response:', weatherData);
    console.log('Weather data inserted into PostgreSQL');
    return res.json(weatherData);
  } catch (error) {
    console.error('Error fetching weather data:', error);
    return res.status(500).json({ error: 'Failed to fetch weather data' });
  }
});

let forecastCache = { data: {}, lastFetch: {} };

app.post('/get-forecast', async (req, res) => {
  const cacheDuration = 2 * 60 * 60 * 1000; // 2 hours in milliseconds
  const now = Date.now();
  const { lat, lon, placeName } = req.body;
  const cacheKey = `${lat},${lon}`;

  if (forecastCache.data[cacheKey] && (now - forecastCache.lastFetch[cacheKey]) < cacheDuration) {
    console.log('Returning cached forecast data for', cacheKey);
    return res.json(forecastCache.data[cacheKey]);
  }

  const requestBody = {
    lat: lat,
    lon: lon,
    model: "gfs",
    parameters: weatherParameters,
    levels: ["surface"],
    key: WINDY_API_KEY
  };

  try {
    const response = await fetch('https://api.windy.com/api/point-forecast/v2', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    const data = await response.json();
    console.log("Windy API forecast response:", data);

    if (data.error) {
      console.error("API Error:", data.error);
      return res.status(400).json({ error: data.error });
    }

    if (!data["temp-surface"] || !data.ts) {
      return res.status(400).json({ error: "Incomplete forecast data" });
    }

    let forecast = [];
    for (let i = 0; i < data["temp-surface"].length; i++) {
      const cloudCover = (
        (data["lclouds-surface"]?.[i] || 0) +
        (data["mclouds-surface"]?.[i] || 0) +
        (data["hclouds-surface"]?.[i] || 0)
      ) / 3;

      forecast.push({
        timestamp: data.ts[i],
        temp: data["temp-surface"][i] - 273.15,
        dewpoint: data["dewpoint-surface"] ? data["dewpoint-surface"][i] - 273.15 : null,
        wind: data["wind-surface"] ? data["wind-surface"][i] : null,
        windGust: data["windGust-surface"] ? data["windGust-surface"][i] : null,
        precip: data["precip-surface"] ? data["precip-surface"][i] : null,
        pressure: data["pressure-surface"] ? data["pressure-surface"][i] : null,
        humidity: data["rh-surface"] ? data["rh-surface"][i] : null,
        cloudCover: cloudCover
      });
    }

    forecastCache.data[cacheKey] = forecast;
    forecastCache.lastFetch[cacheKey] = now;

    return res.json(forecast);
  } catch (error) {
    console.error('Error fetching forecast data:', error);
    return res.status(500).json({ error: 'Failed to fetch forecast data' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

wss.on('connection', (ws) => {
  console.log('Client connected via WebSocket');

  ws.on('close', () => {
    console.log('Client disconnected from WebSocket');
  });
});

// Automated fetch + Postgres insert every 60 sec
setInterval(async () => {
  try {
    const lat = 28.62103;
    const lon = 77.21365;
    const placeName = 'Delhi';
    const weatherData = await fetchAndStore(lat, lon, placeName);

    wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify(weatherData));
      }
    });

    console.log("Sending WebSocket data:", weatherData);
  } catch (error) {
    console.error("Error in automated weather fetch:", error);
  }
}, 60000);

// Forecast broadcast every 2 hours
setInterval(async () => {
  try {
    const entries = Object.entries(forecastCache.data);
    for (const [key, forecastArray] of entries) {
      const forecastToSend = forecastArray.slice(0, 10);
      const payload = {
        type: 'forecast',
        location: key,
        data: forecastToSend
      };

      wss.clients.forEach(client => {
        if (client.readyState === 1) {
          client.send(JSON.stringify(payload));
        }
      });

      console.log(`\n=== Forecast Data for ${key} ===`);
      forecastToSend.forEach(item => {
        const date = new Date(item.timestamp * 1000);
        console.log(`${date.toLocaleString()}: Temp = ${item.temp.toFixed(2)}°C, Wind = ${item.wind} m/s`);
      });
    }
  } catch (error) {
    console.error("Error broadcasting forecast via WebSocket:", error);
  }
}, 2 * 60 * 60 * 1000);