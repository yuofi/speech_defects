from pydantic import BaseModel

class UserRegistration(BaseModel):
    login: str
    password: str

class UserLoginForm(BaseModel):
    login: str
    password: str