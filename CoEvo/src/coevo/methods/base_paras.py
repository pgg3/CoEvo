import os

class Paras:
    def __init__(self, output_dir, **kwargs):
        """
                Initializes the Paras class with the specified parameters.

                Parameters:
                -----------
                output_dir : str
                    The directory where the output will be saved.
                """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)