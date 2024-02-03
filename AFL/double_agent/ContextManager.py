import threading


class NoContextException(Exception):
    pass


class ContextManager:
    """Inherited by Pipeline to allow for context manager abuse

    See https://stackoverflow.com/questions/49573131/how-are-pymc3-variables-assigned-to-the-currently-active-model
    """
    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise NoContextException("No context on context stack")
