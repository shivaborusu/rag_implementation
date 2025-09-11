from qdrant_client.http.models import Distance, VectorParams

def collection_exists(qdrant_client, collection_name):
    return qdrant_client.collection_exists(collection_name=collection_name)
        

def create_collection(qdrant_client, collection_name):
    qdrant_client.create_collection(collection_name=collection_name,
                                    vectors_config=VectorParams(size=1024,
                                                                distance=Distance.COSINE))
    
    return "Success"
