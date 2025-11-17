
import os
import numpy as np
import re

class Hurdat:
    def __init__(self,length,plot):
        self.length=length
        self.dt= 1
        self.q = 1
        self.r = 1
        self.obs_dim = 4
        self.filePath = os.path.join('data','hurdat2-1851-2024-040425.txt')
        self.plot=plot
    
    def h_fn(self,x):
        return x
    
    def R_inv(self,resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def _is_header_line(self, line: str) -> bool:
        """Return True if the line looks like a storm header (starts with ALxxxx)."""
        if not line:
            return False
        # Header lines usually start with 'AL' followed by digits and a comma
        return bool(re.match(r'^\s*AL\d{6}\s*,', line, flags=re.IGNORECASE))

    def sliding_windows(self, arr, win_len, overlap=0):
        step = win_len - overlap
        seq_len = arr.shape[0]
        windows = []

        for start in range(0, seq_len, step):
            end = start + win_len
            if end <= seq_len:
                windows.append(arr[start:end])
            else:
                # pad last window
                pad = np.full((win_len, arr.shape[1]), np.nan, dtype=arr.dtype)
                chunk = arr[start:]
                pad[:chunk.shape[0]] = chunk
                windows.append(pad)
                break

        return windows

    def parse(self):
        """
        Parse HURDAT2 file.
        Returns a list of numpy arrays (seq_len, 4) for each storm (no noise added here).
        """
        storms = []
        with open(self.filePath, "r") as f:
            raw_lines = f.readlines()

        # strip and keep non-empty lines
        lines = [ln.rstrip("\n") for ln in raw_lines if ln.strip() != ""]

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            line = lines[i].strip()

            # if this line isn't a header for some reason, advance to next and warn
            if not self._is_header_line(line):
                print(f"[SKIP/UNEXPECTED LINE] not a header at file line {i}: {line}")
                i += 1
                continue

            # ---- header parsing ----
            header_line = line
            # extract storm id and number of obs with regex to be robust about whitespace/trailing comma
            try:
                # storm id is the first token before comma
                storm_id = header_line.split(",")[0].strip()
                # find the last integer in header (num obs)
                m = re.search(r',\s*([A-Za-z0-9_ -]+?)\s*,\s*(\d+)\s*,?$', header_line)
                if m:
                    storm_name = m.group(1).strip()
                    num_obs = int(m.group(2))
                else:
                    # fallback: split and take last non-empty token that's digits
                    parts = [p.strip() for p in header_line.split(",") if p.strip() != ""]
                    # last part should be num_obs
                    num_obs = int(parts[-1])
                    storm_name = parts[1] if len(parts) > 1 else ""
            except Exception as e:
                print(f"[HEADER PARSE ERROR] file line {i}: {header_line!r}  -- {e}")
                i += 1
                continue

            obs_list = []
            i += 1  # move to first observation line
            obs_read = 0
            while obs_read < num_obs and i < total_lines:
                obs_raw = lines[i].strip()
                # If we hit another header unexpectedly, break
                if self._is_header_line(obs_raw):
                    print(f"[UNEXPECTED HEADER DURING OBS READ] storm {storm_id} expected {num_obs} obs but header found at line {i}: {obs_raw}")
                    break

                try:
                    cols = [c.strip() for c in obs_raw.split(",")]

                    # HURDAT2 convention: date, time, record_id, status, lat, lon, wind, pressure, ...
                    # lat -> cols[4], lon -> cols[5], wind -> cols[6], pressure -> cols[7]
                    if len(cols) < 8:
                        raise ValueError(f"not enough columns ({len(cols)})")

                    lat_str = cols[4]
                    lon_str = cols[5]
                    wind_str = cols[6]
                    pressure_str = cols[7]

                    # handle missing lat/lon (empty strings)
                    if lat_str == "" or lon_str == "":
                        raise ValueError("missing lat/lon")

                    # lat like '28.0N', lon like ' 94.8W'
                    lat_dir = lat_str[-1].upper()
                    lon_dir = lon_str[-1].upper()
                    lat_val = float(lat_str[:-1])
                    lon_val = float(lon_str[:-1])

                    lat = lat_val if lat_dir == "N" else -lat_val
                    lon = lon_val if lon_dir == "E" else -lon_val

                    # convert wind/pressure, handle -999 as NaN
                    wind = float(wind_str) if wind_str not in ("-999", "") else np.nan
                    pressure = float(pressure_str) if pressure_str not in ("-999", "") else np.nan

                    obs_list.append([lat, lon, wind, pressure])
                    obs_read += 1

                except Exception as e:
                    print(f"[OBS PARSE ERROR] storm {storm_id} file line {i}: {obs_raw!r}  -- {e}")
                    # skip this observation but continue reading the rest
                finally:
                    i += 1

            if len(obs_list) > 0:
                storms.append(np.array(obs_list, dtype=np.float32))
            else:
                print(f"[NO VALID OBS] storm {storm_id} had 0 valid observations, skipping.")

            # continue loop (i already points to next line after the observations)
        return storms
    
    def generate(self):
        """Return an array shaped (seq_len, 4) with noise added."""
        storms = self.parse()
        all_windows = []
        plotted = False
        
        for arr in storms:
            if arr.shape[0] < 2:
                continue  


            noise = np.random.normal(0, self.r, size=arr.shape)
            noisy = arr + noise

            windows = self.sliding_windows(noisy, self.length, 2)

            for window in windows:
                if self.plot and not plotted:
                    print(window)
                    import matplotlib.pyplot as plt
                    lat = window[:, 0]
                    lon = window[:, 1]

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)

                  
                    for y in range(-60, 90, 30):
                        ax.plot([-180, 180], [y, y], color="lightgray", linewidth=0.5)

                    for x in range(-180, 210, 30):
                        ax.plot([x, x], [-90, 90], color="lightgray", linewidth=0.5)

                    ax.plot(lon, lat, "-o", markersize=3)

                    ax.set_title("Storm Track (Lat/Lon)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                    pad = 5
                    ax.set_xlim(np.nanmin(lon) - pad, np.nanmax(lon) + pad)
                    ax.set_ylim(np.nanmin(lat) - pad, np.nanmax(lat) + pad)

                    plt.tight_layout()
                    plt.show()
                    plotted = True  

            all_windows.extend(windows)

        return all_windows
            