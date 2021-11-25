class Inference:
    def __init__(self, model):
        self.model = model
        self.model = self.model.cuda()
        self.model.eval()

    def infer_image(self, image_path, **kwargs):
        raise NotImplementedError
