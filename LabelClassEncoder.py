class LabelCountEncoder(object):
    def __init__(self):
        self.count_dict = {}

    def fit(self, column):
        # This gives you a dictionary with level as the key and counts as the value
        count = column.value_counts()
        # We want to rank the key by its value and use the rank as the new value
        self.count_dict = dict(zip(count.index,range(len(count),0,-1)))


    def transform(self, column):
        # If a category only appears in the test set, we will assign the value to zero.
        return column.map(lambda x:self.count_dict.get(x, 0))


    def fit_transform(self, column):
        self.fit(column)
        return self.transform(column)
