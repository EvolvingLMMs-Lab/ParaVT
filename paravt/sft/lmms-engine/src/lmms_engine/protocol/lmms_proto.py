from typing import Protocol


class Runnable(Protocol):
    def run(self, *args, **kwargs):
        ...

    def build(self, *args, **kwargs):
        ...


class Processable(Protocol):
    def process(self, *args, **kwargs):
        ...
