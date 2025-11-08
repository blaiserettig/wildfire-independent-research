"""
Script to train and evaluate the IMPROVED wildfire impact prediction model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from predict_impacts import WildfireImpactPredictor
except ImportError:
    print("ERROR: Could not import improved predictor.")
    raise

def visualize_model_performance(predictor):
    if not hasattr(predictor, 'results'):
        print("No results available for visualization")
        return
    
    results = predictor.results
    
    valid_results = {k: v for k, v in results.items() if v.get('n_test', 0) > 0}
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    model_names = list(valid_results.keys())
    
    r2_scores = [valid_results[m].get('r2', 0) for m in model_names]
    axes[0].barh(model_names, r2_scores, color='steelblue')
    axes[0].set_xlabel('R^2 Score')
    axes[0].set_title('Model Performance - R^2 (Raw Scale)')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='No skill')
    axes[0].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].legend()
    
    r2_log_scores = [valid_results[m].get('r2_log', 0) for m in model_names]
    axes[1].barh(model_names, r2_log_scores, color='darkseagreen')
    axes[1].set_xlabel('R^2 Score (Log Scale)')
    axes[1].set_title('Model Performance - R^2 (Log Scale)')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(0.5, color='green', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')

    rmse_scores = [valid_results[m].get('rmse', 0) for m in model_names]
    axes[2].barh(model_names, rmse_scores, color='coral')
    axes[2].set_xlabel('RMSE (log scale)')
    axes[2].set_title('Root Mean Squared Error')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3, axis='x')

    for i, (val) in enumerate(zip(model_names, rmse_scores)):
        axes[2].text(val * 1.1, i, f'{val:.1e}', va='center', fontsize=8)

    train_sizes = [valid_results[m].get('n_train', 0) for m in model_names]
    test_sizes = [valid_results[m].get('n_test', 0) for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[3].bar(x - width/2, train_sizes, width, label='Training', color='steelblue')
    axes[3].bar(x + width/2, test_sizes, width, label='Test', color='coral')
    axes[3].set_xlabel('Model')
    axes[3].set_ylabel('Sample Size')
    axes[3].set_title('Training vs Test Samples')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(model_names, rotation=45, ha='right')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3, axis='y')

    axes[4].axis('off')
 
    mae_data = []
    for m in model_names:
        mae_val = valid_results[m].get('mae', 0)
        mae_data.append([m.replace('_', ' ').title(), f'{mae_val:,.1f}'])
    
    table = axes[4].table(cellText=mae_data, colLabels=['Model', 'MAE'],
                          cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[4].set_title('Mean Absolute Error by Model', pad=20, fontsize=12, weight='bold')

    for i in range(len(mae_data) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#4472C4')
            table[(i, 1)].set_facecolor('#4472C4')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
            table[(i, 1)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    axes[5].axis('off')
    
    comparison_data = []
    for m in model_names:
        actual = valid_results[m].get('mean_actual', 0)
        predicted = valid_results[m].get('mean_predicted', 0)
        diff_pct = ((predicted - actual) / actual * 100) if actual > 0 else 0
        comparison_data.append([
            m.replace('_', ' ').title(),
            f'{actual:,.0f}',
            f'{predicted:,.0f}',
            f'{diff_pct:+.1f}%'
        ])
    
    table2 = axes[5].table(cellText=comparison_data,
                           colLabels=['Model', 'Actual Mean', 'Predicted Mean', 'Diff %'],
                           cellLoc='left', loc='center',
                           colWidths=[0.4, 0.2, 0.2, 0.2])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    axes[5].set_title('Actual vs Predicted Means', pad=20, fontsize=12, weight='bold')
    
    for i in range(len(comparison_data) + 1):
        for j in range(4):
            if i == 0:
                table2[(i, j)].set_facecolor('#4472C4')
                table2[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table2[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    plt.savefig('img/model_performance_improved.png', dpi=300, bbox_inches='tight')
    print("Model performance visualization saved to img/model_performance_improved.png")
    plt.close()

def create_prediction_examples(predictor):
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS FOR DIFFERENT SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Small Fire, Low Risk',
            'fire_size_ha': 50,
            'fire_location': {'latitude': 47.6, 'longitude': -122.3},  # Seattle
            'drought_index': 100,
            'atmospheric_conditions': {'temperature': 20.0, 'precipitation': 50.0, 'wind_speed': 3.0},
            'svi': 0.3,
            'proximity_to_infrastructure': 50.0,
            'year': 2023,
            'month': 6,
            'state': 'WA'
        },
        {
            'name': 'Medium Fire, Moderate Risk',
            'fire_size_ha': 1000,
            'fire_location': {'latitude': 43.6, 'longitude': -116.2},  # Boise
            'drought_index': 250,
            'atmospheric_conditions': {'temperature': 28.0, 'precipitation': 20.0, 'wind_speed': 6.0},
            'svi': 0.5,
            'proximity_to_infrastructure': 25.0,
            'year': 2023,
            'month': 7,
            'state': 'ID'
        },
        {
            'name': 'Large Fire, High Risk',
            'fire_size_ha': 10000,
            'fire_location': {'latitude': 44.0, 'longitude': -121.5},  # Central Or
            'drought_index': 500,
            'atmospheric_conditions': {'temperature': 35.0, 'precipitation': 5.0, 'wind_speed': 15.0},
            'svi': 0.8,
            'proximity_to_infrastructure': 5.0,
            'year': 2023,
            'month': 8,
            'state': 'OR'
        },
        {
            'name': 'Very Large Fire, Extreme Risk',
            'fire_size_ha': 50000,
            'fire_location': {'latitude': 45.2, 'longitude': -120.8},  # Eastern Or
            'drought_index': 600,
            'atmospheric_conditions': {'temperature': 40.0, 'precipitation': 2.0, 'wind_speed': 20.0},
            'svi': 0.7,
            'proximity_to_infrastructure': 3.0,
            'year': 2023,
            'month': 8,
            'state': 'OR'
        }
    ]
    
    results_table = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ({scenario['fire_size_ha']} ha) ---")

        scenario_for_predict = {k: v for k, v in scenario.items() if k != 'name'}
        prediction = predictor.predict(**scenario_for_predict)
        
        print("Raw Predictions:")
        print(f"  Personnel: {prediction['resource_demand_raw']:.0f}")
        print(f"  Evacuated: {prediction['evacuation_risk_raw']:.0f}")
        print(f"  Structures: {prediction['structure_threat_raw']:.0f}")
        print(f"  Cost: ${prediction['suppression_cost_raw']:,.0f}")
        
        print("Impact Indices (0-1):")
        print(f"  Resource demand: {prediction['resource_demand_index']:.3f}")
        print(f"  Evacuation risk: {prediction['evacuation_risk_index']:.3f}")
        print(f"  Structure threat: {prediction['structure_threat_index']:.3f}")
        print(f"  Suppression cost: {prediction['suppression_cost_index']:.3f}")
        
        results_table.append({
            'Scenario': scenario['name'],
            'Size (ha)': scenario['fire_size_ha'],
            'Personnel': prediction['resource_demand_raw'],
            'Evacuated': prediction['evacuation_risk_raw'],
            'Structures': prediction['structure_threat_raw'],
            'Cost ($)': prediction['suppression_cost_raw'],
            'resource_demand_index': prediction['resource_demand_index'],
            'evacuation_risk_index': prediction['evacuation_risk_index'],
            'structure_threat_index': prediction['structure_threat_index'],
            'suppression_cost_index': prediction['suppression_cost_index']
        })

    results_df = pd.DataFrame(results_table)
    
    axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # impact indices by scenario
    ax = axes[0, 0]
    impact_types = [col for col in results_df.columns if col.endswith('_index')]
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, impact_type in enumerate(impact_types):
        offset = (i - len(impact_types)/2) * width + width/2
        label = impact_type.replace('_index', '').replace('_', ' ').title()
        ax.bar(x + offset, results_df[impact_type], width, label=label)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Impact Index (0-1)')
    ax.set_title('Predicted Impact Indices by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Scenario'], rotation=15, ha='right', fontsize=8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # personnel needs
    ax = axes[0, 1]
    bars = ax.barh(results_df['Scenario'], results_df['Personnel'], color='steelblue')
    ax.set_xlabel('Personnel Needed')
    ax.set_title('Predicted Firefighting Personnel')
    ax.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center', fontsize=8)
    
    # evac numbers
    ax = axes[1, 0]
    bars = ax.barh(results_df['Scenario'], results_df['Evacuated'], color='coral')
    ax.set_xlabel('People Evacuated')
    ax.set_title('Predicted Evacuations')
    ax.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center', fontsize=8)
    
    # suppression costs
    ax = axes[1, 1]
    bars = ax.barh(results_df['Scenario'], results_df['Cost ($)'] / 1e6, color='mediumseagreen')
    ax.set_xlabel('Suppression Cost (Millions $)')
    ax.set_title('Predicted Suppression Costs')
    ax.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'${width:.1f}M', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('img/prediction_examples_improved.png', dpi=300, bbox_inches='tight')
    print("\nPrediction examples visualization saved to img/prediction_examples_improved.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("WILDFIRE IMPACT PREDICTION MODEL")
    print("Training & Evaluation")
    print("="*60)
    
    predictor = WildfireImpactPredictor()

    print("\nTraining models...")
    results = predictor.train(test_size=0.2, random_state=42)

    print("\nSaving model...")
    predictor.save('models/wildfire_impact_predictor_improved.pkl')

    print("\nCreating performance visualizations...")
    visualize_model_performance(predictor)

    print("\nGenerating example predictions...")
    create_prediction_examples(predictor)
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*60)
    print("\nModel saved to: models/wildfire_impact_predictor_improved.pkl")
    print("Visualizations saved to: img/")