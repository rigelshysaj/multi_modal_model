import re


def get_image_filename(description):
    # Lowercase the description
    filename = description.lower()
    # Replace spaces and hyphens with underscores
    filename = filename.replace(' ', '_').replace('-', '_')
    # Remove any characters that are not lowercase letters, numbers, or underscores
    filename = re.sub(r'[^a-z0-9_]', '', filename)
    return filename