class CCCDData:
    def __init__(self):
        self.name = None
        self.id = None
        self.dob = None
        self.gender = None
        self.origin_place = None
        self.current_place = None

    def __repr__(self):
        return str(self.__dict__)
