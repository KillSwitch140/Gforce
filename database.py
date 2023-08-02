import pymongo

# Function to create a MongoDB connection
def create_mongodb_connection(database_name):
    try:
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = mongo_client[database_name]
        return db
    except pymongo.errors.ConnectionFailure as e:
        print(e)
        return None

# Create the resumes collection in MongoDB
def create_resumes_collection(db):
    if db is not None:
        db.create_collection("resumes")

# Function to store resume and information in MongoDB
def insert_resume(resumes_collection, candidate_info):
    resumes_collection.insert_one(candidate_info)

# Function to retrieve all resumes from MongoDB
def get_all_resumes(resumes_collection):
    return list(resumes_collection.find())

