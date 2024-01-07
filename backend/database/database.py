from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field
from pymongo import MongoClient

# Provide the mongodb atlas url to connect python to mongodb using pymongo
CONNECTION_STRING = "mongodb://root:example@mongodb:27017"
#
# Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
client = MongoClient(CONNECTION_STRING)


class OID(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            return ObjectId(str(v))
        except Exception:
            raise ValueError("Not a valid ObjectId")


class MongoModel(BaseModel):
    id: Optional[OID] = Field(None, alias="_id")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: lambda oid: str(oid),
        }

    @classmethod
    def from_mongo(cls, data: dict):
        id = data.pop("_id", None)
        return cls(**dict(data, id=id))

    def mongo(self, **kwargs):
        parsed = self.model_dump(**kwargs)
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = parsed.pop("id")
        return parsed
