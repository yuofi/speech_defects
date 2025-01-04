import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    username = Column(String(50), nullable=False)

class TokenTable(Base):
    __tablename__ = "token"
    user_id = Column(Integer)
    access_toke = Column(String(450), primary_key=True)
    refresh_toke = Column(String(450),nullable=False)
    status = Column(Boolean)
    created_date = Column(DateTime, default=datetime.datetime.now)

class Progress(Base):
    __tablename__ = "progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_name = Column(String(100), nullable=False)  # Имя курса
    completed_tasks = Column(Integer, default=0)  # Количество выполненных заданий
    total_tasks = Column(Integer, nullable=False)  # Общее количество заданий в курсе
    last_updated = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

