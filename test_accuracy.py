#!/usr/bin/env python3

import json
from coordinate_accuracy_tester import CoordinateAccuracyTester

# Load the results
with open('lot2_test_results.json', 'r') as f:
    results = json.load(f)

# Test accuracy
tester = CoordinateAccuracyTester()
accuracy_results = tester.run_comprehensive_test(results, results.get('extracted_text', ''), 'LOT 2 324 Dolan Rd Aerial Map.pdf')

print('=== COORDINATE ACCURACY ANALYSIS ===')
for test_name, result in accuracy_results['tests'].items():
    print(f'{test_name}: {result["score"]:.3f} ({"PASS" if result["passed"] else "FAIL"})')
    if result.get('details'):
        print(f'  Details: {result["details"]}')
    if result.get('issues'):
        for issue in result['issues']:
            print(f'  Issue: {issue}')

print(f'\nOverall Score: {accuracy_results["overall_accuracy"]["weighted_score"]:.3f}')
print(f'Grade: {accuracy_results["overall_accuracy"]["grade"]}')
print(f'\nRecommendations:')
for rec in accuracy_results['recommendations']:
    print(f'- {rec}') 