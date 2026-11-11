"""Bind author-generated reader IDs to the external image IDs by exact path."""


def bind_reader_ids(logical_images, reader_paths):
    """Return native reader ID to external image ID for an exact one-to-one roster."""
    expected = dict((path, image_id) for image_id, path in logical_images)
    unique_ids = {image_id for image_id, _ in logical_images}
    if (len(expected) != len(logical_images) or len(unique_ids) != len(logical_images)
            or len(reader_paths) != len(logical_images) or set(reader_paths.values()) != set(expected)):
        raise ValueError("Reader paths and logical image IDs must identify exactly one visit per image")
    return {reader_id: expected[path] for reader_id, path in reader_paths.items()}
