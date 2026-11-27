from . import _DatasetID

class ObjInfo:
    addr: int

def get_info(loc: _DatasetID) -> ObjInfo: ...
