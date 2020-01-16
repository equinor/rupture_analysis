from typing import Dict

import pyodbc


class Singleton(type):
    _instances: Dict[str, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MSSQLConnectionFactory(metaclass=Singleton):
    _client: pyodbc.Connection = None

    def get_client(self, conn_str: str) -> pyodbc.Connection:
        if not self._client:
            self._connect(conn_str)
        return self._client

    def _connect(self, conn_str: str) -> pyodbc.Connection:
        conn = pyodbc.connect(conn_str)
        self._client = conn
