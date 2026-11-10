"""Observe exact stock Paddle CPU geometry in a separate class namespace."""
from paddle_geometry import load_stock, transform_batch


class StockTrace:
    """Replay the original methods while recording only their observed arguments."""

    def __init__(self, source_dir):
        self.stock = load_stock(source_dir)
        self.steps = []
        self.resize_index = 0
        classes = self.stock.operators
        decode, pad = classes["Decode"].apply, classes["Pad"].apply_image
        crop = classes["RandomCrop"]._crop_image
        flip, resize = classes["RandomFlip"].apply_image, classes["Resize"].apply_image

        def observe_decode(operator, sample, context=None):
            output = decode(operator,sample,context=context)
            self.steps.append([])
            return output

        def observe_pad(operator,image,offsets,im_size,size):
            output = pad(operator,image,offsets,im_size,size)
            if tuple(im_size) != image.shape[:2] or tuple(size) != output.shape[:2]:
                raise ValueError("Stock expansion shape witness")
            self.steps[-1].append(("pad",image.shape[:2],output.shape[:2],tuple(map(int,offsets))))
            return output

        def observe_crop(operator,image,window):
            output = crop(operator,image,window)
            self.steps[-1].append(("crop",image.shape[:2],output.shape[:2],tuple(map(int,window))))
            return output

        def observe_flip(operator,image):
            output = flip(operator,image)
            self.steps[-1].append(("flip",image.shape[:2],output.shape[:2],()))
            return output

        def observe_resize(operator,image,scale):
            output = resize(operator,image,scale)
            self.steps[self.resize_index].append(("resize",image.shape[:2],output.shape[:2],tuple(map(float,scale))))
            self.resize_index += 1
            return output

        classes["Decode"].apply = observe_decode
        classes["Pad"].apply_image = observe_pad
        classes["RandomCrop"]._crop_image = observe_crop
        classes["RandomFlip"].apply_image = observe_flip
        classes["Resize"].apply_image = observe_resize

    def run(self,samples,*,seed):
        self.steps = []
        self.resize_index = 0
        views = transform_batch(self.stock,samples,seed=seed)
        if not len(self.steps)==self.resize_index==len(views):
            raise ValueError("Expected one observed resize per original view")
        for steps,view in zip(self.steps,views,strict=True):
            kinds = [step[0] for step in steps]
            expected = [kind for kind,present in (("pad",view.expanded),("crop",view.cropped),("flip",view.flipped))
                        if present]+["resize"]
            if (kinds!=expected or steps[0][1]!=view.source_hw
                or steps[-1][1]!=view.before_resize["image"].shape[:2]
                or steps[-1][2]!=view.output["image"].shape[-2:]):
                raise ValueError("Observed path disagrees with original stock fields")
        return views,self.steps

