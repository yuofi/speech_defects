from peewee import *

connection = SqliteDatabase('database.db')



class BaseModel(Model):
    class Meta:
        database = connection

class User(BaseModel):
    user_id = AutoField()
    login = CharField(unique=True)
    password = CharField()

    class Meta:
        db_table = 'Users'
        order_by = ('user_id',)


class Course(BaseModel):
    course_id = AutoField()
    name = CharField()
    progress = IntegerField()

    class Meta:
        db_table = 'Courses'
        order_by = ('course_id',)
from peewee import *

connection = SqliteDatabase('database.db')



class BaseModel(Model):
    class Meta:
        database = connection

class User(BaseModel):
    user_id = AutoField()
    login = CharField(unique=True)
    password = CharField()

    class Meta:
        db_table = 'Users'
        order_by = ('user_id',)


class Course(BaseModel):
    course_id = AutoField()
    name = CharField()
    progress = IntegerField()

    class Meta:
        db_table = 'Courses'
        order_by = ('course_id',)
