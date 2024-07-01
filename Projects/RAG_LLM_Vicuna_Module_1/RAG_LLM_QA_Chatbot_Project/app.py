import urllib


class LocalTunnelSetup:
    def __init__(self, port=8000, subdomain="aivn-simple-rag"):
        self.port = port
        self.subdomain = subdomain

    def print_ip(self):
        ip = urllib.request.urlopen(
            'https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
        print("Password/Endpoint IP for localtunnel is:", ip)

    def run_localtunnel(self):
        import os
        os.system(f'lt --port {self.port} --subdomain {self.subdomain}')


if __name__ == "__main__":
    setup = LocalTunnelSetup()
    setup.print_ip()
    setup.run_localtunnel()
