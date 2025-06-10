from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class student(BaseModel):
    name:str='AK' # here 'AK' is default values
    age : Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,le=10)

new_student={'name':'Anjani Kumar','age':'43','email':'anjani@gmail.com','cgpa':10} #test1

#new_student={} #test2

result=student(**new_student)

#print(result.name)
print(result)

#convert pydantic object to dict

student_result=dict(result)

print(student_result)