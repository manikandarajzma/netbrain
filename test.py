import requests
import time
import urllib3
import pprint
import json
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import netbrainauth
# Set the request inputs
token = netbrainauth.get_auth_token()
nb_url = "http://localhost"
full_url = nb_url + "/ServicesAPI/API/V1/CMDB/Path/Calculation"
headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
headers["Token"] = token

sourceIP = "10.210.1.0"
sourcePort = 0
sourceGwIP = "10.210.1.0"
sourceGwDev = "GW2Lab"
sourceGwIntf =  "GigabitEthernet4"
destIP = "10.210.1.0"
destPort = 0
pathAnalysisSet = 1
protocol = 4
isLive = False

body = {
            "sourceIP" : sourceIP,                # IP address of the source device.
            "sourcePort" : sourcePort,
            "sourceGwDev" : sourceGwDev,          # Hostname of the gateway device.
            "sourceGwIP" : sourceGwIP,            # Ip address of the gateway device.
            "sourceGwIntf" : sourceGwIntf,        # Name of the gateway interface.
            "destIP" : destIP,                    # IP address of the destination device.
            "destPort" : destPort,
            "pathAnalysisSet" : pathAnalysisSet,  # 1:L3 Path; 2:L2 Path; 3:L3 Active Path
            "protocol" : protocol,                # Specify the application protocol, check online help, such as 4 for IPv4.
            "isLive" : isLive                     # False: Current Baseline; True: Live access
    } 

try:
    response = requests.post(full_url, data = json.dumps(body), headers = headers, verify = False)
    if response.status_code == 200:
        result = response.json()
        print (result)
    else:
        print ("Create module attribute failed! - " + str(response.text))
    
except Exception as e:
    print (str(e)) 