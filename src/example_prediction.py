"""
Simple example script showing how to use the trained wildfire impact prediction model.
"""

from predict_impacts import WildfireImpactPredictor
import os

def main():
    print("="*60)
    print("WILDFIRE IMPACT PREDICTION - EXAMPLE USAGE")
    print("="*60)

    predictor = WildfireImpactPredictor()

    model_path = 'models\wildfire_impact_predictor_improved.pkl'
    if os.path.exists(model_path):
        print(f"\nLoading trained model from {model_path}...")
        predictor.load(model_path)
    else:
        print(f"\nModel not found at {model_path}")
        print("Training new model...")
        predictor.train()
        predictor.save(model_path)
    
    # Ex1: high-risk
    print("\n" + "-"*60)
    print("EXAMPLE 1: High-Risk")
    print("-"*60)
    print("Size: Very large")
    print("Location: Central Oregon")
    print("Conditions: Extreme drought, high temperatures, strong winds")
    print("Proximity: Close to energy infrastructure")
    print("Vulnerability: High SVI")
    
    prediction1 = predictor.predict(
        fire_size_ha=10000,
        fire_location={'latitude': 44.0, 'longitude': -121.5},
        drought_index=500,
        atmospheric_conditions={
            'temperature': 35.0, 
            'precipitation': 5.0,
            'wind_speed': 15.0 
        },
        svi=0.8,
        proximity_to_infrastructure=5.0, 
        year=2023,
        month=8,
        state='OR'
    )
    
    max_value = max(prediction1.values())
    print("\nPredicted Impact Indices:")
    for key, value in sorted(prediction1.items()):
        normalized = value / max_value if max_value else 0
        bar_length = int(normalized * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {key:30s} {value:.3f} {bar}")

    
    # Ex2: low-risk
    print("\n" + "-"*60)
    print("EXAMPLE 2: Low-Risk Fire")
    print("-"*60)
    print("Size: Small")
    print("Location: Western Washington")
    print("Conditions: Low drought, moderate temperatures, light winds")
    print("Proximity: Far from energy infrastructure")
    print("Vulnerability: Low SVI")
    
    prediction2 = predictor.predict(
        fire_size_ha=400,
        fire_location={'latitude': 47.6, 'longitude': -122.3},
        drought_index=100,
        atmospheric_conditions={
            'temperature': 20.0,
            'precipitation': 50.0,
            'wind_speed': 3.0
        },
        svi=0.3,
        proximity_to_infrastructure=50.0,
        year=2023,
        month=6,
        state='WA'
    )
    
    print("\nPredicted Impact Indices:")
    max_value = max(prediction2.values())
    for key, value in sorted(prediction2.items()):
        normalized = value / max_value if max_value else 0
        bar_length = int(normalized * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {key:30s} {value:.3f} {bar}")
    
    # Ex3: defaults
    print("\n" + "-"*60)
    print("EXAMPLE 3: Minimal Input (Using Defaults)")
    print("-"*60)
    print("Only location provided - other parameters use defaults")
    
    prediction3 = predictor.predict(
        fire_size_ha=1000,
        fire_location=(43.6, -116.2),  # Boise
        state='ID'
    )
    
    print("\nPredicted Impact Indices (with defaults):")
    max_value = max(prediction3.values())
    for key, value in sorted(prediction3.items()):
        normalized = value / max_value if max_value else 0
        bar_length = int(normalized * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {key:30s} {value:.3f} {bar}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

