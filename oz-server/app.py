import hug

@hug.get()
def health():
    """checks status of application"""
    return "OK"
