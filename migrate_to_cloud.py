from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_URL = "http://localhost:6333"
CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
CLOUD_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION = "saudi_vision_2030"
BATCH_SIZE = 100

print("Connecting to local Qdrant...")
local_client = QdrantClient(url=LOCAL_URL)

print("Connecting to Qdrant Cloud...")
cloud_client = QdrantClient(
    url=CLOUD_URL,
    api_key=CLOUD_KEY,
)

local_info = local_client.get_collection(COLLECTION)

print("Recreating collection in Qdrant Cloud to match local schema...")
# Delete the improperly formatted cloud collection if it exists
try:
    cloud_client.delete_collection(collection_name=COLLECTION)
    print("Old cloud collection deleted.")
except Exception:
    pass

# Recreate it using the exact configuration from the local collection
cloud_client.create_collection(
    collection_name=COLLECTION,
    vectors_config=local_info.config.params.vectors,
    sparse_vectors_config=local_info.config.params.sparse_vectors,
)
print("Collection successfully recreated in cloud!")

print("Starting migration in batches...")
offset = None
total_migrated = 0

while True:
    records, next_offset = local_client.scroll(
        collection_name=COLLECTION,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        break
        
    points_to_upload = [
        PointStruct(id=record.id, vector=record.vector, payload=record.payload)
        for record in records
    ]

    cloud_client.upsert(
        collection_name=COLLECTION,
        points=points_to_upload,
    )
    total_migrated += len(records)
    print(f"Migrated {total_migrated} points...")

    if next_offset is None:
        break
    offset = next_offset

cloud_info = cloud_client.get_collection(COLLECTION)
print(f"\nMigration complete!")
print(f"Cloud points: {cloud_info.points_count}")
print(f"Expected: {local_info.points_count}")
if cloud_info.points_count == local_info.points_count:
    print("VERIFICATION PASSED - all vectors migrated successfully")
else:
    print("WARNING - point count mismatch, check migration")