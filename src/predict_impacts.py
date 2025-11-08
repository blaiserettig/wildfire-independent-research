"""
Machine learning model to predict downstream impacts of wildfire events.

This module provides functionality to predict impact indices for:
1. Resource demand (personnel needed)
2. Evacuation risk (people displaced)
3. Structure threat (buildings at risk)
4. Suppression cost

"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xarray as xr
import rasterio
import warnings
warnings.filterwarnings('ignore')

class WildfireImpactPredictor:
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.target_scalers = {}
        self.feature_names = []
        self.trained = False
        
    def load_all_data(self):
        print("Loading data sources...")

        print("  Loading MTBS data...")
        self.fires = gpd.read_file(f"{self.data_dir}/mtbs_perimeter_data/mtbs_perims_DD_pnw.shp")
        self.fires['Ig_Date'] = pd.to_datetime(self.fires['Ig_Date'], errors='coerce')
        self.fires['YEAR'] = self.fires['Ig_Date'].dt.year
        self.fires['MONTH'] = self.fires['Ig_Date'].dt.month
        self.fires['STATE'] = self.fires['Event_ID'].str[:2]
        self.fires = self.fires.to_crs(epsg=5070)
        self.fires['area_ha'] = self.fires.geometry.area / 10000.0
        self.fires = self.fires[(self.fires['YEAR'] >= 2000) & (self.fires['YEAR'] <= 2024)]

        self.fires_geo = self.fires.to_crs(epsg=4326)
        self.fires_geo['centroid_lon'] = self.fires_geo.geometry.centroid.x
        self.fires_geo['centroid_lat'] = self.fires_geo.geometry.centroid.y

        print("  Loading ICS209 data...")
        self.ics = pd.read_csv(f"{self.data_dir}/ics209/ics209-plus-wf_sitreps_1999to2020.csv", low_memory=False)
        self.ics['REPORT_TO_DATE'] = pd.to_datetime(self.ics['REPORT_TO_DATE'], format='%m/%d/%Y', errors='coerce')
        self.ics['DISCOVERY_DATE'] = pd.to_datetime(self.ics['DISCOVERY_DATE'], format='%m/%d/%Y', errors='coerce')
        self.ics_pnw = self.ics[self.ics['POO_STATE'].isin(['WA', 'OR', 'ID'])].copy()
        
        self.ics_agg = self.ics_pnw.groupby('FIRE_EVENT_ID').agg({ # agg ics209 by fire event
            'ACRES': 'max',
            'TOTAL_PERSONNEL': 'max',
            'NUM_EVACUATED': 'max',
            'STR_DESTROYED': 'max',
            'STR_THREATENED': 'max',
            'EST_IM_COST_TO_DATE': 'max',
            'POO_STATE': 'first',
            'CY': 'first'
        }).reset_index()
        self.ics_agg['YEAR'] = pd.to_numeric(self.ics_agg['CY'], errors='coerce')
        
        print(f"  Total ICS209 records: {len(self.ics_agg)}")
        self.ics_with_impacts = self.ics_agg[
            (self.ics_agg['TOTAL_PERSONNEL'] > 0) |
            (self.ics_agg['NUM_EVACUATED'] > 0) |
            (self.ics_agg['STR_THREATENED'] > 0) |
            (self.ics_agg['STR_DESTROYED'] > 0)
        ].copy()
        print(f"  Records with impact data: {len(self.ics_with_impacts)}")
        
        print("  Joining MTBS fires with ICS209 impact data...")

        print(f"    Sample MTBS Event_IDs: {self.fires_geo['Event_ID'].head(3).tolist()}")
        print(f"    Sample ICS209 FIRE_EVENT_IDs: {self.ics_with_impacts['FIRE_EVENT_ID'].head(3).tolist()}")

        self.fires_geo = self.fires_geo.merge(
            self.ics_with_impacts,
            left_on='Event_ID',
            right_on='FIRE_EVENT_ID',
            how='inner',
            suffixes=('', '_ics')
        )
        print(f"  Fires matched by Event_ID: {len(self.fires_geo)}")

        if len(self.fires_geo) == 0:
            print("  No direct ID matches. Attempting spatial-temporal matching...")

            self.fires_geo = self.fires.to_crs(epsg=4326).copy()
            self.fires_geo['centroid_lon'] = self.fires_geo.geometry.centroid.x
            self.fires_geo['centroid_lat'] = self.fires_geo.geometry.centroid.y

            fires_for_match = self.fires_geo.copy()
            fires_for_match['MTBS_ACRES'] = fires_for_match['area_ha'] * 2.47105

            ics_for_match = self.ics_with_impacts.copy()
            ics_for_match['STATE'] = ics_for_match['POO_STATE']

            matched_fires = []
            
            for idx, fire in fires_for_match.iterrows():
                candidates = ics_for_match[
                    (ics_for_match['STATE'] == fire['STATE']) & 
                    (ics_for_match['YEAR'] == fire['YEAR'])
                ]
                
                if len(candidates) > 0:
                    candidates = candidates.copy()
                    candidates['size_diff'] = abs(candidates['ACRES'] - fire['MTBS_ACRES'])
                    candidates['size_ratio'] = candidates['ACRES'] / (fire['MTBS_ACRES'] + 1)

                    candidates = candidates[
                        (candidates['size_ratio'] > 0.5) & 
                        (candidates['size_ratio'] < 2.0)
                    ]
                    
                    if len(candidates) > 0:
                        best_match = candidates.nsmallest(1, 'size_diff').iloc[0]

                        fire_with_ics = fire.to_dict()
                        fire_with_ics.update({
                            'FIRE_EVENT_ID': best_match['FIRE_EVENT_ID'],
                            'TOTAL_PERSONNEL': best_match['TOTAL_PERSONNEL'],
                            'NUM_EVACUATED': best_match['NUM_EVACUATED'],
                            'STR_DESTROYED': best_match['STR_DESTROYED'],
                            'STR_THREATENED': best_match['STR_THREATENED'],
                            'EST_IM_COST_TO_DATE': best_match['EST_IM_COST_TO_DATE'],
                            'ACRES': best_match['ACRES']
                        })
                        matched_fires.append(fire_with_ics)
            
            if len(matched_fires) > 0:
                self.fires_geo = gpd.GeoDataFrame(matched_fires, crs='EPSG:4326')
                print(f"  Fires matched by spatial-temporal matching: {len(self.fires_geo)}")
            else:
                print("  ERROR: No fires could be matched with ICS209 data")
                
        print(f"  Final dataset size: {len(self.fires_geo)} fires with impact data")

        print("  Loading DSCI data...")
        self.dsci_df = pd.read_csv(f"{self.data_dir}/dsci/dm_export_20000101_20241231.csv")
        self.dsci_df['MapDate'] = pd.to_datetime(self.dsci_df['MapDate'], format='%Y%m%d', errors='coerce')
        self.dsci_df['YEAR'] = self.dsci_df['MapDate'].dt.year
        self.dsci_df['MONTH'] = self.dsci_df['MapDate'].dt.month
        self.dsci_df['STATE'] = self.dsci_df['Name'].map({'Oregon': 'OR', 'Washington': 'WA', 'Idaho': 'ID'})
        self.dsci_pnw = self.dsci_df[
            self.dsci_df['STATE'].isin(['WA', 'OR', 'ID']) & 
            (self.dsci_df['YEAR'] >= 2000) & (self.dsci_df['YEAR'] <= 2024)
        ].copy()
        
        print("  Loading SVI data...")
        self.svi_data = {}
        pnw_fips = {'WA': '53', 'OR': '41', 'ID': '16'}
        svi_years = [2000, 2010, 2014, 2016, 2018, 2020, 2022]
        
        for year in svi_years:
            svi_file = f"{self.data_dir}/svi/SVI_{year}_US_county.csv"
            if os.path.exists(svi_file):
                try:
                    df = pd.read_csv(svi_file, low_memory=False)
                    if 'STATE_FIPS' in df.columns:
                        df_pnw = df[df['STATE_FIPS'].isin(pnw_fips.values())].copy()
                        if len(df_pnw) > 0:
                            df_pnw['FIPS'] = df_pnw['STCOFIPS'].astype(str).str.zfill(5)
                            df_pnw['STATE_ABBR'] = df_pnw.get('STATE_ABBR', '')
                            df_pnw['RPL_THEMES'] = df_pnw.get('RPL_THEMES', df_pnw.get('USTP', np.nan))
                            df_pnw['E_TOTPOP'] = df_pnw.get('E_TOTPOP', df_pnw.get('Totpop2000', np.nan))
                            df_pnw['YEAR'] = year
                            self.svi_data[year] = df_pnw
                except Exception as e:
                    print(f"    Warning: Could not load SVI {year}: {e}")

        print("  Loading energy infrastructure data...")
        self.plants_gdf = None
        try:
            elec_data = []
            pnw_states = ['USA-WA', 'USA-OR', 'USA-ID']
            
            with open(f"{self.data_dir}/eia/ELEC/ELEC.txt", "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get('geography', '').startswith('USA-') and any(
                            state in record.get('geography', '') for state in pnw_states
                        ):
                            elec_data.append(record)
                    except json.JSONDecodeError:
                        continue
            
            plant_info = []
            for record in elec_data:
                if record.get('f') == 'A' and 'GEN' in record.get('series_id', ''):
                    try:
                        geography = record.get('geography', '')
                        state = geography.split('-')[1] if '-' in geography else None
                        lat = float(record.get('lat', 0)) if record.get('lat') else None
                        lon = float(record.get('lon', 0)) if record.get('lon') else None
                        
                        if lat and lon and lat != 0 and lon != 0:
                            plant_info.append({
                                'series_id': record.get('series_id', ''),
                                'plant_name': record.get('name', '').split(':')[1].strip() if ':' in record.get('name', '') else record.get('name', ''),
                                'state': state,
                                'lat': lat,
                                'lon': lon
                            })
                    except (ValueError, KeyError, IndexError):
                        continue
            
            if len(plant_info) > 0:
                plants_df = pd.DataFrame(plant_info).drop_duplicates(subset=['series_id', 'lat', 'lon'])
                self.plants_gdf = gpd.GeoDataFrame(
                    plants_df,
                    geometry=gpd.points_from_xy(plants_df['lon'], plants_df['lat']),
                    crs='EPSG:4326'
                )
                self.plants_gdf = self.plants_gdf.to_crs('EPSG:5070')
                print(f"    Loaded {len(self.plants_gdf)} power plants")
        except Exception as e:
            print(f"    Error: Could not load energy data: {e}")
        
        print("  Setting up climate data paths...")
        def prism_raster_paths(base_dir, variable, years=range(2000, 2025)):
            paths = {}
            for y in years:
                if variable == 'tmean':
                    paths[y] = f"{base_dir}/mean_temp/annual/prism_tmean_us_25m_{y}/prism_tmean_us_25m_{y}.tif"
                elif variable == 'ppt':
                    paths[y] = f"{base_dir}/precip/annual/prism_ppt_us_25m_{y}/prism_ppt_us_25m_{y}.tif"
            return paths

        self.prism_paths = {
            'tmean': prism_raster_paths("data/prism", 'tmean'),
            'ppt': prism_raster_paths("data/prism", 'ppt')
        }
        
        print("Data loading complete!")
        print(f"Final dataset size: {len(self.fires_geo)} fires with impact data")
        
    def extract_climate_features(self, fires_df):
        print("Extracting climate features...")
        
        fires_with_climate = fires_df.copy()
        fires_with_climate['tmean'] = np.nan
        fires_with_climate['ppt'] = np.nan
        fires_with_climate['wind_speed'] = np.nan
        
        for idx, fire in fires_with_climate.iterrows():
            year = int(fire['YEAR'])
            
            if year in self.prism_paths['tmean'] and year in self.prism_paths['ppt']:
                try:
                    with rasterio.open(self.prism_paths['tmean'][year]) as src:
                        tmean_vals = [x[0] for x in src.sample([(fire['centroid_lon'], fire['centroid_lat'])])]
                        if tmean_vals and tmean_vals[0] > -9000:
                            fires_with_climate.loc[idx, 'tmean'] = tmean_vals[0]
                    
                    with rasterio.open(self.prism_paths['ppt'][year]) as src:
                        ppt_vals = [x[0] for x in src.sample([(fire['centroid_lon'], fire['centroid_lat'])])]
                        if ppt_vals and ppt_vals[0] > -9000:
                            fires_with_climate.loc[idx, 'ppt'] = ppt_vals[0]
                except Exception as e:
                    pass

            try:
                nc_file = f"{self.data_dir}/terra/ws/TerraClimate_ws_{year}.nc"
                if os.path.exists(nc_file):
                    ds = xr.open_dataset(nc_file)
                    month = fires_with_climate.loc[idx, 'MONTH'] if 'MONTH' in fires_with_climate.columns and pd.notna(fires_with_climate.loc[idx, 'MONTH']) else 7
                    month = int(month) if pd.notna(month) else 7
                    month = max(1, min(12, month))
                    ws_data = ds.sel(lon=fire['centroid_lon'], lat=fire['centroid_lat'], method='nearest')
                    ws_value = ws_data['ws'].isel(time=month-1).values if month <= 12 else ws_data['ws'].isel(time=6).values
                    fires_with_climate.loc[idx, 'wind_speed'] = float(ws_value)
                    ds.close()
            except Exception as e:
                pass
        
        return fires_with_climate
    
    def extract_dsci_features(self, fires_df):
        fires_with_dsci = fires_df.copy()

        dsci_annual = self.dsci_pnw.groupby(['STATE', 'YEAR']).agg({
            'DSCI': ['mean', 'max', 'std']
        }).reset_index()
        dsci_annual.columns = ['STATE', 'YEAR', 'dsci_mean', 'dsci_max', 'dsci_std']
        
        fires_with_dsci = fires_with_dsci.merge(
            dsci_annual,
            on=['STATE', 'YEAR'],
            how='left'
        )

        if 'MONTH' in fires_with_dsci.columns:
            fires_with_dsci['fire_month'] = fires_with_dsci['MONTH']
        elif 'Ig_Date' in fires_with_dsci.columns:
            fires_with_dsci['fire_month'] = pd.to_datetime(fires_with_dsci['Ig_Date'], errors='coerce').dt.month.fillna(7)
        else:
            fires_with_dsci['fire_month'] = 7
        
        dsci_monthly = self.dsci_pnw.groupby(['STATE', 'YEAR', 'MONTH']).agg({
            'DSCI': 'mean'
        }).reset_index()
        dsci_monthly.columns = ['STATE', 'YEAR', 'MONTH', 'dsci_monthly']
        
        fires_with_dsci = fires_with_dsci.merge(
            dsci_monthly,
            left_on=['STATE', 'YEAR', 'fire_month'],
            right_on=['STATE', 'YEAR', 'MONTH'],
            how='left'
        )
        
        return fires_with_dsci
    
    def extract_infrastructure_features(self, fires_df):
        fires_with_infra = fires_df.copy()
        
        if self.plants_gdf is None or len(self.plants_gdf) == 0:
            fires_with_infra['nearest_plant_dist_km'] = np.nan
            fires_with_infra['plants_within_10km'] = 0
            fires_with_infra['plants_within_50km'] = 0
            return fires_with_infra
        
        fires_for_proximity = fires_df[['Event_ID', 'geometry']].copy()
        fires_for_proximity = fires_for_proximity.to_crs('EPSG:5070')
        
        fire_plant_distances = []
        for idx, fire in fires_for_proximity.iterrows():
            distances = self.plants_gdf.geometry.distance(fire.geometry)
            nearest_dist_km = distances.min() / 1000
            plants_10km = (distances / 1000 < 10).sum()
            plants_50km = (distances / 1000 < 50).sum()
            
            fire_plant_distances.append({
                'Event_ID': fire['Event_ID'],
                'nearest_plant_dist_km': nearest_dist_km,
                'plants_within_10km': plants_10km,
                'plants_within_50km': plants_50km
            })
        
        proximity_df = pd.DataFrame(fire_plant_distances)
        fires_with_infra = fires_with_infra.merge(proximity_df, on='Event_ID', how='left')
        
        return fires_with_infra
    
    def extract_svi_features(self, fires_df):
        fires_with_svi = fires_df.copy()

        svi_state_avg = {}
        for year, df in self.svi_data.items():
            if 'RPL_THEMES' in df.columns and 'STATE_ABBR' in df.columns:
                state_avg = df.groupby('STATE_ABBR')['RPL_THEMES'].mean().to_dict()
                for state, val in state_avg.items():
                    if state not in svi_state_avg:
                        svi_state_avg[state] = []
                    svi_state_avg[state].append(val)
        
        for state in svi_state_avg:
            svi_state_avg[state] = np.mean(svi_state_avg[state]) if svi_state_avg[state] else np.nan
        
        fires_with_svi['svi_rank'] = fires_with_svi['STATE'].map(svi_state_avg)
        
        return fires_with_svi
    
    def create_target_variables(self, fires_df):
        targets = fires_df[['Event_ID']].copy()

        targets['resource_demand'] = fires_df['TOTAL_PERSONNEL'].fillna(0)

        targets['evacuation_risk'] = fires_df['NUM_EVACUATED'].fillna(0)

        targets['structure_threat'] = (
            fires_df['STR_THREATENED'].fillna(0) + 
            fires_df['STR_DESTROYED'].fillna(0)
        )

        targets['suppression_cost'] = fires_df['EST_IM_COST_TO_DATE'].fillna(0)
        
        return targets
    
    def build_features(self, fires_df):
        print("Building features...")
        
        required_cols = ['Event_ID', 'YEAR', 'STATE']
        missing_cols = [col for col in required_cols if col not in fires_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        features = fires_df[required_cols].copy()

        features['fire_size_ha'] = fires_df['area_ha']
        features['log_fire_size'] = np.log1p(fires_df['area_ha'])

        if 'MONTH' in fires_df.columns:
            features['MONTH'] = fires_df['MONTH']
        elif 'Ig_Date' in fires_df.columns:
            features['MONTH'] = pd.to_datetime(fires_df['Ig_Date'], errors='coerce').dt.month.fillna(7)
        else:
            features['MONTH'] = 7

        features['latitude'] = fires_df['centroid_lat']
        features['longitude'] = fires_df['centroid_lon']

        features = pd.get_dummies(features, columns=['STATE'], prefix='state')

        features['month_sin'] = np.sin(2 * np.pi * features['MONTH'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['MONTH'] / 12)

        if 'tmean' in fires_df.columns:
            features['tmean'] = fires_df['tmean']
        if 'ppt' in fires_df.columns:
            features['ppt'] = fires_df['ppt']
        if 'wind_speed' in fires_df.columns:
            features['wind_speed'] = fires_df['wind_speed']

        if 'dsci_mean' in fires_df.columns:
            features['dsci_mean'] = fires_df['dsci_mean']
        if 'dsci_monthly' in fires_df.columns:
            features['dsci_monthly'] = fires_df['dsci_monthly']

        if 'nearest_plant_dist_km' in fires_df.columns:
            features['nearest_plant_dist_km'] = fires_df['nearest_plant_dist_km']
            features['plants_within_10km'] = fires_df['plants_within_10km']
            features['plants_within_50km'] = fires_df['plants_within_50km']

        if 'svi_rank' in fires_df.columns:
            features['svi_rank'] = fires_df['svi_rank']

        if 'tmean' in features.columns and 'ppt' in features.columns:
            features['temp_precip_ratio'] = features['tmean'] / (features['ppt'] + 1)
            features['drought_index'] = features['tmean'] * (1 / (features['ppt'] + 1))

        if 'tmean' in features.columns:
            features['size_temp_interaction'] = features['log_fire_size'] * features['tmean']
        if 'dsci_mean' in features.columns:
            features['size_drought_interaction'] = features['log_fire_size'] * features['dsci_mean']
        if 'wind_speed' in features.columns:
            features['size_wind_interaction'] = features['log_fire_size'] * features['wind_speed']
        
        self.feature_names = [col for col in features.columns if col != 'Event_ID']
        
        return features
    
    def train(self, test_size=0.2, random_state=42):
        print("\n" + "="*60)
        print("TRAINING WILDFIRE IMPACT PREDICTION MODELS")
        print("="*60)

        self.load_all_data()
        
        if len(self.fires_geo) < 50:
            raise ValueError(f"Insufficient data: only {len(self.fires_geo)} fires with impact data")

        fires_with_features = self.extract_climate_features(self.fires_geo)
        fires_with_features = self.extract_dsci_features(fires_with_features)
        fires_with_features = self.extract_infrastructure_features(fires_with_features)
        fires_with_features = self.extract_svi_features(fires_with_features)
        
        if 'MONTH' not in fires_with_features.columns:
            if 'Ig_Date' in fires_with_features.columns:
                fires_with_features['MONTH'] = pd.to_datetime(fires_with_features['Ig_Date'], errors='coerce').dt.month.fillna(7)
            else:
                fires_with_features['MONTH'] = 7

        X = self.build_features(fires_with_features)
  
        targets = self.create_target_variables(fires_with_features)
  
        merged = X.merge(targets, on='Event_ID', how='inner')

        feature_cols = [col for col in merged.columns if col not in 
                       ['Event_ID', 'resource_demand', 'evacuation_risk', 
                        'structure_threat', 'suppression_cost']]
        X_final = merged[feature_cols].fillna(0)
        X_final = X_final.replace([np.inf, -np.inf], np.nan).fillna(0)
 
        target_names = ['resource_demand', 'evacuation_risk', 'structure_threat', 'suppression_cost']

        y_targets = merged[target_names]

        X_train, X_test, y_train, y_test = train_test_split(
            X_final, 
            y_targets,
            test_size=test_size,
            random_state=random_state
        )
 
        y_train_dict = {name: y_train[name].values for name in target_names}
        y_test_dict = {name: y_test[name].values for name in target_names}

        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)

        results = {}
        
        for target_name in target_names:
            print(f"\n--- Training model for: {target_name} ---")
            
            y_train = y_train_dict[target_name]
            y_test = y_test_dict[target_name]

            valid_train_mask = (y_train > 0) & np.isfinite(y_train)
            valid_test_mask = (y_test > 0) & np.isfinite(y_test)
            
            n_valid_train = valid_train_mask.sum()
            n_valid_test = valid_test_mask.sum()
            
            print(f"  Valid training samples: {n_valid_train} / {len(y_train)}")
            print(f"  Valid test samples: {n_valid_test} / {len(y_test)}")
            
            if n_valid_train < 10:
                print(f"  Warning: Very few valid training samples")
                valid_train_mask = np.ones(len(y_train), dtype=bool)
            
            X_train_valid = X_train_scaled[valid_train_mask]
            y_train_valid = y_train[valid_train_mask]

            y_train_log = np.log1p(y_train_valid)

            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                subsample=0.8
            )
            
            model.fit(X_train_valid, y_train_log)
            self.models[target_name] = model

            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)

            y_train_pred_log = model.predict(X_train_scaled[valid_train_mask])
            y_train_pred = np.expm1(y_train_pred_log)
            
            self.target_scalers[target_name] = MinMaxScaler()
            all_values = np.concatenate([y_train_valid, y_train_pred, y_test, y_pred])
            all_values = all_values[all_values > 0]
            if len(all_values) > 0:
                self.target_scalers[target_name].fit(all_values.reshape(-1, 1))
            else:
                self.target_scalers[target_name].fit([[0], [1]])

            if n_valid_test > 0:
                mae = mean_absolute_error(y_test[valid_test_mask], y_pred[valid_test_mask])
                rmse = np.sqrt(mean_squared_error(y_test[valid_test_mask], y_pred[valid_test_mask]))
                r2 = r2_score(y_test[valid_test_mask], y_pred[valid_test_mask])

                y_test_log = np.log1p(y_test[valid_test_mask])
                y_pred_log_test = np.log1p(y_pred[valid_test_mask])
                r2_log = r2_score(y_test_log, y_pred_log_test)
                
                results[target_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'r2_log': r2_log,
                    'n_train': n_valid_train,
                    'n_test': n_valid_test,
                    'mean_actual': y_test[valid_test_mask].mean(),
                    'mean_predicted': y_pred[valid_test_mask].mean()
                }
                
                print(f"  MAE: {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  R^2 (raw): {r2:.3f}")
                print(f"  R^2 (log scale): {r2_log:.3f}")
                print(f"  Mean actual: {y_test[valid_test_mask].mean():.2f}")
                print(f"  Mean predicted: {y_pred[valid_test_mask].mean():.2f}")
            else:
                print(f"  No valid test samples for evaluation")
                results[target_name] = {'n_train': n_valid_train, 'n_test': 0}
        
        self.trained = True
        self.results = results
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nModels trained on {len(X_train)} samples with {len(feature_cols)} features")
        
        return results
    
    def predict(self, fire_size_ha, fire_location, drought_index=None, atmospheric_conditions=None, 
                svi=None, proximity_to_infrastructure=None, year=None, month=None, state=None):
        """
        Predict impact indices for a new fire event.
        
        Parameters:
        -----------
        fire_size_ha : float
            Fire size in hectares
        fire_location : dict or tuple
            {'latitude': float, 'longitude': float} or (lat, lon)
        drought_index : float, optional
            DSCI
        atmospheric_conditions : dict, optional
            {'temperature': float, 'precipitation': float, 'wind_speed': float}
        svi : float, optional
            Social Vulnerability Index rank (0-1)
        proximity_to_infrastructure : float, optional
            Distance to nearest power plant in km
        year : int, optional
            Year of fire
        month : int, optional
            Month of fire
        state : str, optional
            State code
        
        Returns:
        --------
        dict : Normalized impact indices (0-1) and raw predictions for:
            - resource_demand (personnel)
            - evacuation_risk (people)
            - structure_threat (buildings)
            - suppression_cost (dollars)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(fire_location, (tuple, list)):
            lat, lon = fire_location
        else:
            lat = fire_location.get('latitude', fire_location.get('lat'))
            lon = fire_location.get('longitude', fire_location.get('lon'))

        features = {}

        features['fire_size_ha'] = fire_size_ha
        features['log_fire_size'] = np.log1p(fire_size_ha)

        features['latitude'] = lat
        features['longitude'] = lon

        if state:
            for state_code in ['WA', 'OR', 'ID']:
                features[f'state_{state_code}'] = 1 if state_code == state else 0
        else:
            for state_code in ['WA', 'OR', 'ID']:
                features[f'state_{state_code}'] = 0

        if month:
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)
        else:
            features['month_sin'] = 0
            features['month_cos'] = 0

        if atmospheric_conditions:
            features['tmean'] = atmospheric_conditions.get('temperature', 0)
            features['ppt'] = atmospheric_conditions.get('precipitation', 0)
            features['wind_speed'] = atmospheric_conditions.get('wind_speed', 0)
        else:
            features['tmean'] = 0
            features['ppt'] = 0
            features['wind_speed'] = 0

        if drought_index is not None:
            features['dsci_mean'] = drought_index
            features['dsci_monthly'] = drought_index
        else:
            features['dsci_mean'] = 0
            features['dsci_monthly'] = 0

        if proximity_to_infrastructure is not None:
            features['nearest_plant_dist_km'] = proximity_to_infrastructure
            features['plants_within_10km'] = 1 if proximity_to_infrastructure < 10 else 0
            features['plants_within_50km'] = 1 if proximity_to_infrastructure < 50 else 0
        else:
            features['nearest_plant_dist_km'] = 100
            features['plants_within_10km'] = 0
            features['plants_within_50km'] = 0

        if svi is not None:
            features['svi_rank'] = svi
        else:
            features['svi_rank'] = 0.5

        if features['tmean'] > 0 and features['ppt'] > 0:
            features['temp_precip_ratio'] = features['tmean'] / (features['ppt'] + 1)
            features['drought_index'] = features['tmean'] * (1 / (features['ppt'] + 1))
        else:
            features['temp_precip_ratio'] = 0
            features['drought_index'] = 0

        if features['tmean'] > 0:
            features['size_temp_interaction'] = features['log_fire_size'] * features['tmean']
        else:
            features['size_temp_interaction'] = 0
        
        if features['dsci_mean'] > 0:
            features['size_drought_interaction'] = features['log_fire_size'] * features['dsci_mean']
        else:
            features['size_drought_interaction'] = 0
        
        if features['wind_speed'] > 0:
            features['size_wind_interaction'] = features['log_fire_size'] * features['wind_speed']
        else:
            features['size_wind_interaction'] = 0

        feature_df = pd.DataFrame([features])
        
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[self.feature_names]

        X_scaled = self.scalers['features'].transform(feature_df)

        predictions = {}
        indices = {}
        
        for target_name in ['resource_demand', 'evacuation_risk', 'structure_threat', 'suppression_cost']:
            y_pred_log = self.models[target_name].predict(X_scaled)[0]
            y_pred = np.expm1(y_pred_log)
            
            y_pred_scaled = self.target_scalers[target_name].transform([[max(0, y_pred)]])[0][0]
            y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
            
            predictions[f'{target_name}_raw'] = y_pred
            indices[f'{target_name}_index'] = y_pred_scaled

        return {**predictions, **indices}
    
    def save(self, filepath='models/wildfire_impact_predictor_improved.pkl'):
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'target_scalers': self.target_scalers,
                'feature_names': self.feature_names,
                'trained': self.trained,
                'results': getattr(self, 'results', {})
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/wildfire_impact_predictor_improved.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.scalers = data['scalers']
        self.target_scalers = data['target_scalers']
        self.feature_names = data['feature_names']
        self.trained = data['trained']
        self.results = data.get('results', {})
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Creating and training Wildfire Impact Predictor...")
    
    predictor = WildfireImpactPredictor()
    results = predictor.train()

    predictor.save()
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    print("\n--- Small Fire (100 ha) ---")
    small_fire = predictor.predict(
        fire_size_ha=100,
        fire_location={'latitude': 45.5, 'longitude': -122.5},
        drought_index=350,
        atmospheric_conditions={'temperature': 25.0, 'precipitation': 10.0, 'wind_speed': 8.0},
        svi=0.6,
        proximity_to_infrastructure=15.0,
        year=2023,
        month=8,
        state='OR'
    )
    
    print("Raw Predictions:")
    print(f"  Personnel needed: {small_fire['resource_demand_raw']:.0f}")
    print(f"  People evacuated: {small_fire['evacuation_risk_raw']:.0f}")
    print(f"  Structures threatened: {small_fire['structure_threat_raw']:.0f}")
    print(f"  Suppression cost: ${small_fire['suppression_cost_raw']:,.0f}")
    
    print("\nNormalized Indices (0-1):")
    print(f"  Resource demand: {small_fire['resource_demand_index']:.3f}")
    print(f"  Evacuation risk: {small_fire['evacuation_risk_index']:.3f}")
    print(f"  Structure threat: {small_fire['structure_threat_index']:.3f}")
    print(f"  Suppression cost: {small_fire['suppression_cost_index']:.3f}")

    print("\n--- Large Fire (10,000 ha) ---")
    large_fire = predictor.predict(
        fire_size_ha=10000,
        fire_location={'latitude': 44.0, 'longitude': -121.5},
        drought_index=500,
        atmospheric_conditions={'temperature': 35.0, 'precipitation': 5.0, 'wind_speed': 15.0},
        svi=0.8,
        proximity_to_infrastructure=5.0,
        year=2023,
        month=8,
        state='OR'
    )
    
    print("Raw Predictions:")
    print(f"  Personnel needed: {large_fire['resource_demand_raw']:.0f}")
    print(f"  People evacuated: {large_fire['evacuation_risk_raw']:.0f}")
    print(f"  Structures threatened: {large_fire['structure_threat_raw']:.0f}")
    print(f"  Suppression cost: ${large_fire['suppression_cost_raw']:,.0f}")
    
    print("\nNormalized Indices (0-1):")
    print(f"  Resource demand: {large_fire['resource_demand_index']:.3f}")
    print(f"  Evacuation risk: {large_fire['evacuation_risk_index']:.3f}")
    print(f"  Structure threat: {large_fire['structure_threat_index']:.3f}")
    print(f"  Suppression cost: {large_fire['suppression_cost_index']:.3f}")