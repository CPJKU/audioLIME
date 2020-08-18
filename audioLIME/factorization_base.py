class Factorization(object):
    def __init__(self):
        pass

    def compose_model_input(self, components=None):
        raise NotImplementedError

    def get_number_components(self):
        raise NotImplementedError

    def retrieve_components(self, selection_order=None):
        raise NotImplementedError

    def get_ordered_component_names(self): # e.g. instrument names
        raise NotImplementedError

    def set_analysis_window(self, start_sample, y_length):
        raise NotImplementedError