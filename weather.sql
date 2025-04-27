CREATE TABLE weather (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP NOT NULL,
  place_name VARCHAR(255),
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  temp REAL,
  dewpoint REAL,
  wind REAL,
  windgust REAL,
  precip REAL,
  pressure REAL,
  humidity REAL,
  cloudcover REAL
);
