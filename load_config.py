from optparse import OptionParser
import json

def load_config():
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="config", help="specify configuration file")
 
    (options, args) = parser.parse_args()

    if not options.config:
        options.config = 'configuration.json'
    
    config = json.load(open(options.config, 'r'))
    return config