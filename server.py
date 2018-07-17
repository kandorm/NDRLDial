from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from urlparse import urlparse, parse_qs

from utils.commandparser import NDRLDialOptParser
from NDRLDial import NDRLDial

import json
import numpy as np


class DataEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, float):
            return str(o)
        elif isinstance(o, np.float32):
            return str(o)
        else:
            return json.JSONEncoder.default(self, o)


class ChatHandler(BaseHTTPRequestHandler):
    model = None

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Credentials", "false")

    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Access-Control-Allow-Credentials", "false")
        self.send_header('Content-type','application/json')
        self.end_headers()

        q = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        #print q
        response = model.reply(
            q['user_utt'][0], q['last_sys_act'][0],
            q['belief_state'][0])
        self.request.sendall(json.dumps(response, cls=DataEncoder))

    def do_POST(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Access-Control-Allow-Credentials", "false")
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        data = self.rfile.read(int(self.headers['Content-Length']))
        q = json.loads(data)
        #print q
        response = model.reply(
            q['user_utt'], q['last_sys_act'],
            q['belief_state'])
        self.wfile.write(json.dumps(response, cls=DataEncoder))
        self.wfile.close()


if __name__ == '__main__':
    # TODO: IP address and port
    hostname = 'localhost'
    port = 8000

    # loading neural dialog model
    args = NDRLDialOptParser()
    config = args.config
    model = NDRLDial(config)
    model.reply('', '', {})

    # handler
    ChatHandler.model = model

    # launch server
    httpd = HTTPServer((hostname, port), ChatHandler)
    print 'Server is on - %s:%d' % (hostname, port)
    httpd.serve_forever()
