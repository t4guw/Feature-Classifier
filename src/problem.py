import json

# Class for serializing custom objects into JSON. 
# See https://docs.python.org/3/library/json.html#json.JSONEncoder.default
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, P):
            return obj.__dict__
        return json.JSONEncoder(self, obj)

class Problem(object):
    def __init__(self):
        self.number = 1
        self.solutions = []
        self.statement = ""
    
    def serialize(self):
        return json.dumps(self, cls=ComplexEncoder)
