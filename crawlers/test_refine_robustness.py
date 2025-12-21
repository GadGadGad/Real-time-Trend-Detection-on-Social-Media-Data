import json
import os
import sys
import re

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from crawlers.analyze_trends import refine_trends_preprocessing

# Mock trends from user request
trends = {
    'real madrid đấu với man city': {'keywords': [], 'volume': 200000},
    'bảng tổng sắp huy chương 33': {'keywords': [], 'volume': 100000},
    'xổ số miền nam ngày 12 tháng 12': {'keywords': [], 'volume': 500000},
    'epic': {'keywords': [], 'volume': 20000},
}

# Scenario: LLM returns stripped diacritics or partial strings
# This is what often happens with 2B models
refined_data_mock = {
    "filtered": [
        "xo so mien nam ngay 12 thang 12", # Stripped diacritics
        "epic",                           # Exact match
        "xo so"                           # Partial/Category match
    ],
    "merged": {
        "real madrid dau voi man city": "Real vs MC", # Stripped diacritics
        "bang tong sap huy chuong 33": "SEA Games 33" # Stripped diacritics
    }
}

class MockRefiner:
    def __init__(self, *args, **kwargs):
        self.enabled = True
        self.debug = True
    def refine_trends(self, trends):
        return refined_data_mock

# Patch
import crawlers.analyze_trends
crawlers.analyze_trends.LLMRefiner = MockRefiner

print("--- Robustness Test: Vietnamese Diacritics & Partial Matches ---")
result = refine_trends_preprocessing(
    trends, 
    llm_provider="mock", 
    gemini_api_key="none", 
    llm_model_path="none", 
    debug_llm=True
)

print(f"\nResulting trends count: {len(result)}")
for k in result.keys():
    print(f"- {k}")

# Assertions to show current failures
expected_reduction = True
if len(result) == 4:
    print("\n❌ FAILURE: No trends were removed/merged. Case-insensitive exact matching is NOT enough for Vietnamese.")
else:
    print("\n✅ SUCCESS: Refinement worked (partially or fully).")
