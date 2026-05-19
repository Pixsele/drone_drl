import socket
import struct
import threading
import time


class UEDataReceiverUDP:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, ip: str = '127.0.0.1', port: int = 9876):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(ip, port)
        return cls._instance

    def _init(self, ip: str, port: int):
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clientSocket.bind((ip, port))

        self.data = None
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                data, addr = self.clientSocket.recvfrom(1024)
                x, y, z = struct.unpack("fff", data[:12])
                with self.lock:
                    self.data = (x, y, z)
            except socket.timeout:
                continue

    def get_pos(self):
        with self.lock:
            return self.data


class UEDataSenderUDP:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, ip: str = '127.0.0.1', port: int = 9877):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(ip, port)
        return cls._instance

    def _init(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_bool(self, value: bool):
        data = struct.pack("?", value)
        self.sock.sendto(data, (self.ip, self.port))

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    server = UEDataSenderUDP()
    client = UEDataReceiverUDP()

    server.send_bool(True)
    time.sleep(1)
    print(client.get_pos())

    # while True:
    #     server.send_bool(True)
    #     time.sleep(0.1)


