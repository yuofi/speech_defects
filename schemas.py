from pydantic import BaseModel
import datetime

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class requestdetails(BaseModel):
    email:str
    password:str
        
class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str

class changepassword(BaseModel):
    email:str
    old_password:str
    new_password:str

class TokenCreate(BaseModel):
    user_id:str
    access_token:str
    refresh_token:str
    status:bool
    created_date:datetime.datetime
    
class ProgressCreate(BaseModel):
    course_name: str
    completed_tasks: int
    total_tasks: int

class ProgressUpdate(BaseModel):
    course_name: str
    completed_tasks: int

class ProgressResponse(BaseModel):
    course_name: str
    completed_tasks: int
    total_tasks: int
    last_updated: datetime.datetime
