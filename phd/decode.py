#!/usr/bin/env python3
import sys
import math

def displayConfiguration(configuration:str, negate:bool, csv:bool, csvSemicolon:bool):
    if (csv | csvSemicolon):
        if (csvSemicolon):
            configuration = ";".join([*configuration])
        else:
            configuration = ",".join([*configuration])

    print (configuration)
    if (negate):
        configuration = configuration.replace('0','O').replace('1','0').replace('O','1')
        print (configuration)

if (__name__ == "__main__") :   
    import argparse
    import io

    parser = argparse.ArgumentParser(description='Decode binary file containing 1D Cellular Automata binary configurations.')
    parser.add_argument('-N', type=int, help='Expected number of the cells in the configuration.', required=True)
    parser.add_argument('-file', type=str, help='Specify the file name (when not specified, then stdin is used).', required=False)
    parser.add_argument('--negate', help='Return also the negation of the configuration.', action='store_true')
    parser.add_argument('--csv', help='Use comma delimiter to separate cell values.', action='store_true')
    parser.add_argument('--csv-semicolon', help='Use semicolon delimiter to separate cell values.', action='store_true')
    
    args = parser.parse_args()
    
    bytesPerConfiguration = math.ceil(args.N / 8)
    redundantBits = bytesPerConfiguration * 8 - args.N
    
    if (args.file):
        handler = open(args.file, mode='rb')
    else:
        handler = sys.stdin.buffer

    partNumber = 0
    configuration = ""    
    try:
        for chunk in handler.read():
            if (partNumber == bytesPerConfiguration - 1):
                configuration += "{0:08b}".format(chunk)[:-redundantBits]
                displayConfiguration(configuration, args.negate, args.csv, args.csv_semicolon)
                partNumber = 0
                configuration = ""    
            else:
                configuration += "{0:08b}".format(chunk)
                partNumber += 1
    finally:
        if (handler is not None and handler is io.TextIOWrapper):
            print ('CLOSE')
            handler.close()

    if (partNumber != 0):
        raise f"Seems the file does not contain configurations of length {args.N} as still some data are required (reading part {partNumber})."