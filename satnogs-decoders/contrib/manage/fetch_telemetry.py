#!/usr/bin/env python3

import os
import json
import time
import argparse
import binascii
import sys
from datetime import datetime

from satnogs_api_client import fetch_telemetry, DB_BASE_URL, DB_DEV_BASE_URL
from dotenv import load_dotenv
from db_helpers import gridsquare
from os import environ
from qth_locator import square_to_location, location_to_square


load_dotenv()


SATNOGS_DB_API_TOKEN = environ.get('SATNOGS_DB_API_TOKEN', None)
DB_SPUTNIX_BASE_URL = 'http://db.satnogs.sputnix.ru'
REQUIRED_VARIABLES = ["SATNOGS_DB_API_TOKEN"]


def validate_settings():
    settings = globals()
    settings_valid = True

    for variable in REQUIRED_VARIABLES:
        if not settings[variable]:
            print("{} not configured but required.".format(variable))
            settings_valid = False

    if not settings_valid:
        sys.exit(-1)

validate_settings()

def fetch_and_store_telemetry(norad_id, max_frames, source, telemetry_dir):
    if source == 'satnogs':
        url = DB_BASE_URL
    elif source == 'satnogs-dev':
        url = DB_DEV_BASE_URL
    elif source == 'sputnix':
        url = DB_SPUTNIX_BASE_URL

    telemetry = fetch_telemetry(norad_id, max_frames, url, SATNOGS_DB_API_TOKEN)
    print("Fetched {} frames.".format(len(telemetry)))

    directory = os.path.join(telemetry_dir, source, str(norad_id))
    filename = '{:%Y%m%d%H%M%S}_all_telemetry.json'.format(datetime.now())
    path  = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, 'w') as f:
        json.dump(telemetry, f)

    print("Stored in {}".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and store all telemetry data from a satnogs-db instance for a given satellite.')
    parser.add_argument('norad_id', type=int, help='NORAD ID of the satellite')
    parser.add_argument('--source',
                        type=str,
                        default='satnogs',
                        help='satnogs-db Instance: satnogs, satnogs-dev or sputnix')
    parser.add_argument('--base_dir',
                        type=str,
                        default='./telemetry/',
                        help='Base directory of the telemetry storage')
    parser.add_argument('--max',
                        type=int,
                        default=25,
                        help='Maximum number of fetched frames. Default: 25')
    args = parser.parse_args()

    fetch_and_store_telemetry(args.norad_id, args.max, args.source, args.base_dir)
