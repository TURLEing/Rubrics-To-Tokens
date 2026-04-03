# set RAY_DEDUP_LOGS=0 before importing ray
import os
os.environ["RAY_DEDUP_LOGS"] = os.getenv("RAY_DEDUP_LOGS", "0")

try:
    import nebula_patch
except Exception as e:
    import traceback
    print("Error importing nebula_patch: ", e, traceback.format_exc())
