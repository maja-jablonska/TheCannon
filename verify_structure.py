import sys
import os

# Add the parent directory to sys.path to allow importing TheCannon
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Attempting to import TheCannon...")
    import TheCannon
    print("Successfully imported TheCannon.")

    print("Checking exposed classes...")
    print(f"CannonModel: {TheCannon.CannonModel}")
    print(f"Dataset: {TheCannon.Dataset}")
    print(f"diagnostics: {TheCannon.diagnostics}")

    print("Attempting to import surveys...")
    from TheCannon.surveys import apogee, lamost
    print("Successfully imported surveys.apogee and surveys.lamost.")

    print("Verification successful!")

except ImportError as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
