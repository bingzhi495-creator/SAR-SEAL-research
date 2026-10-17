"""UNBOUND injected-Torch image/geometry contract; no numerical execution yet.

RGB uint8 is a new branch input convention, not a reconstruction of teacher
normalization. Tiny canvas/branch sizes support the one analytic public test.
"""


def make_geometry(torch, ori_shape, content_shape, canvas_size=800, branch_size=400):
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    def positive_pair(value, label):
        need(type(value) in (tuple, list) and len(value) == 2
             and all(type(dimension) is int and dimension > 0 for dimension in value),
             label+' must contain two positive integers')
        return tuple(value)

    original = positive_pair(ori_shape, 'ori_shape')
    content = positive_pair(content_shape, 'content_shape')
    need(type(canvas_size) is int and canvas_size > 0
         and type(branch_size) is int and branch_size > 0,
         'canvas_size and branch_size must be positive integers')
    need(max(content) == canvas_size, 'Content must fit the canvas with its longest side equal to it')
    longest = max(original)
    # Exact integer round-half-up of dimension*canvas_size/longest.
    expected = tuple((2*dimension*canvas_size+longest)//(2*longest) for dimension in original)
    need(content == expected, 'Content shape must match original keep-ratio round-half-up geometry')

    class ImageGeometry:
        def prepare_rgb(self, image_uint8):
            need(isinstance(image_uint8, torch.Tensor)
                 and tuple(image_uint8.shape) == (1, 3, original[0], original[1])
                 and image_uint8.dtype == torch.uint8,
                 'One RGB uint8 tensor [1,3,original_H,original_W] is required')
            image = image_uint8.to(dtype=torch.float32)/255.
            resized = torch.nn.functional.interpolate(image, size=content, mode='bilinear',
                align_corners=False, antialias=False)
            canvas = torch.nn.functional.pad(resized,
                (0, canvas_size-content[1], 0, canvas_size-content[0]), mode='constant', value=0.)
            return torch.nn.functional.interpolate(canvas, size=(branch_size, branch_size),
                mode='bilinear', align_corners=False, antialias=False)

        def render_delta(self, delta):
            need(isinstance(delta, torch.Tensor) and delta.ndim == 4 and delta.shape[0] > 0
                 and tuple(delta.shape[1:]) == (1, branch_size, branch_size)
                 and delta.is_floating_point() and bool(torch.isfinite(delta).all()),
                 'Finite floating delta [N,1,branch_size,branch_size] with N>0 is required')
            canvas = torch.nn.functional.interpolate(delta, size=(canvas_size, canvas_size),
                mode='bilinear', align_corners=False, antialias=False)
            cropped = canvas[:, :, :content[0], :content[1]]
            return torch.nn.functional.interpolate(cropped, size=original, mode='bilinear',
                align_corners=False, antialias=False)

    return ImageGeometry()
