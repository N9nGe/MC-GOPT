"""Check if multi-size training is actually using different bin sizes"""
import re

# Parse the log file to see what bin sizes are being used
try:
    with open('/tmp/train_output.log', 'r') as f:
        content = f.read()
        
    # Look for any bin size information
    if 'MultiSizeWrapper' in content:
        print("✓ MultiSizeWrapper is being used")
    else:
        print("✗ MultiSizeWrapper NOT found in logs")
        
    # Check for the observation size warnings (should be NONE with our fix)
    warnings = re.findall(r'WARNING: Observation size mismatch', content)
    print(f"\nObservation size warnings: {len(warnings)}")
    
except FileNotFoundError:
    print("Log file not found - training might not have run long enough")

print("\nTo verify multi-size is working, add debug logging to the wrapper:")
print("In envs/Packing/multiSizeWrapper.py, add to reset():")
print('  print(f"Reset with bin_size: {self.current_bin_size}")')
