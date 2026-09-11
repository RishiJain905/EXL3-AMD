"""Real loopback reservation/restart checks; no model or GPU."""
import os
from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from serve_exl3 import reserve_socket


class SocketLifecycleTests(unittest.TestCase):
    def test_active_listener_stays_exclusive(self):
        with reserve_socket('127.0.0.1', 0) as listener:
            with self.assertRaises(OSError):
                reserve_socket('127.0.0.1', listener.getsockname()[1])

    @unittest.skipIf(os.name == 'nt', 'Backend runs on Linux; Windows uses exclusive binding')
    def test_restart_after_server_closes_established_connection(self):
        with reserve_socket('127.0.0.1', 0) as listener:
            port = listener.getsockname()[1]
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                connection, _ = listener.accept()
                # Server actively closes, leaving its endpoint in TIME_WAIT.
                connection.close()
                self.assertEqual(client.recv(1), b'')
        with reserve_socket('127.0.0.1', port) as restarted:
            self.assertEqual(restarted.getsockname()[1], port)


if __name__ == '__main__':
    unittest.main()
