class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        childern=None,
        value=None,
        is_leaf=False,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.childern = childern if childern is not None else {}
        self.value = value
        self.is_leaf = is_leaf
