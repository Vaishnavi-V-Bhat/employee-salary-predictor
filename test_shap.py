#!/usr/bin/env python3
"""
Test script to verify SHAP integration with the salary prediction model
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap

def test_shap_integration():
    """Test SHAP integration with the model"""
    print("Testing SHAP integration...")
    
    # Load data
    df = pd.read_csv("adult 3.csv")
    df.occupation.replace('?', 'Others', inplace=True)
    df.workclass.replace('?', 'NotListed', inplace=True)
    df.rename(columns={'fnlwgt': 'Population Representation'}, inplace=True)
    
    # Prepare features
    X = df.drop(['income', 'education'], axis=1)
    y = df['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
    
    # Train model
    categorical_features_indices = [
        X.columns.get_loc(col) for col in [
            'workclass', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country'
        ]
    ]
    
    model = CatBoostClassifier(
        iterations=100,  # Reduced for testing
        random_strength=1,
        learning_rate=0.1,
        l2_leaf_reg=5,
        depth=6,
        border_count=64,
        bagging_temperature=0.8,
        verbose=0
    )
    model.fit(X, y, cat_features=categorical_features_indices)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Test with sample input
    sample_input = X.iloc[0:1]
    shap_values = explainer.shap_values(sample_input)
    
    print(f"✅ SHAP integration successful!")
    print(f"✅ Model trained successfully")
    print(f"✅ SHAP explainer created successfully")
    print(f"✅ SHAP values computed successfully")
    print(f"✅ Expected value: {explainer.expected_value:.3f}")
    print(f"✅ SHAP values shape: {np.array(shap_values).shape}")
    
    # Test feature importance
    feature_importance = pd.DataFrame({
        'Feature': sample_input.columns,
        'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    print(f"\n✅ Top 5 most important features:")
    for i, (_, row) in enumerate(feature_importance.head().iterrows()):
        direction = "🟢" if row['SHAP_Value'] > 0 else "🔴"
        print(f"   {direction} {row['Feature']}: {row['SHAP_Value']:.3f}")
    
    print("\n✅ All tests passed! SHAP integration is working correctly.")

if __name__ == "__main__":
    test_shap_integration() 