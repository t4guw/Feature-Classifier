import json

# Class for serializing custom objects into JSON. 
# See https://docs.python.org/3/library/json.html#json.JSONEncoder.default
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Problem):
            return obj.__dict__
        return json.JSONEncoder(self, obj)

# Class that defines the properties of each Problem object
class Problem(object):
    def __init__(self, number, statement):
        self.number = number
        self.statement = statement
        self.solutions = []
    
    def serialize(self):
        return json.dumps(self, cls=ComplexEncoder)
