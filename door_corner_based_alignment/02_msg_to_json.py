'''
##usage
python3 msg2json.py {input_msgfile_directory} {output_jsonfile_directory}
'''

import sys
import msgpack
import json

def main(bin_fn, dest_fn):

    #Read file as binary and unpack data using MessagePack Library
    with open(bin_fn,"rb") as f:
        data = msgpack.unpackb(f.read(), use_list = False,raw=False)
    #Write point coordinates into file, one point for one line
    with open(dest_fn,"w") as json_file:
        json.dump(data,json_file,indent=2)
            
    print("Finished")

if __name__=="__main__":
    argv = sys.argv

    if len(argv) <3 :
        print("Read al content in the map file and dump into a json file")
        print("Usage :")
        print("python msg_to_json.py [msg file] [json destination]")

    else:
        bin_fn = argv[1]
        dest_fn = argv[2]
        main(bin_fn, dest_fn)
