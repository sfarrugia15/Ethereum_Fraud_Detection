# Created on 19.02.2019 By Steven Farrugia
# Scope: Retrieve and analyze fraudulent activity on the Ethereum network
# Method of retrieval: EtherscamDB API; https://etherscamdb.info/api, MyEtherWallet; https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/addresses/addresses-darklist.json
import datetime
import json
import requests


"""
Scope: Get all documented scams using etherscamDB API
Method Info: Parse response as json file and retain relevant fields on which analysis shall be carried out
Total Number of Available documented scams: 6434 (19/02/2019)
"""
def get_scams():
    response = requests.get("https://etherscamdb.info/api/scams/")
    if response.status_code == 200:
        response = response.json()
        print(response['result'][0]['name'])
        no_of_scams = len(response['result'])
        scam_id, scam_name, scam_status, scam_category = ([] for i in range(4))

        for scam in range(no_of_scams):
            scam_id.append(response['result'][scam]['id'])
            scam_name.append(response['result'][scam]['name'])
            scam_status.append(response['result'][scam]['status'])
            # if 'category' not in response['result'][scam]:
            #     scam_category.append(response['result'][scam]['category'])
            # else:
            #     scam_category.append('Null')
            # print (response['result'][scam]['category'])

    print(scam_id[1], " ", scam_name[1], " ", scam_status[1])


"""
Scope: Get all documented addresses from EtherscamDB API
Method Info: Parse response as json file and retain addresses and scam IDs
Total Number of Available documented scam addresses: 1988 (19/02/2019)
"""
def get_scam_addresses():
    response = requests.get("https://etherscamdb.info/api/addresses/")
    if response.status_code == 200:
        response = response.json()
        scam_id, scam_address = ([] for i in range(2))

        for scam in response['result']:
            scam_address.append(scam)
            scam_id.append(response['result'][scam]['id'])

        print(scam_address[0], ' ', scam_id[0])


"""
Scope: Additional documented scam/illicit behavior addresses
Method Info: Parse local json file and return as array
Total Number of Available documented scam addresses: 692 (19/02/2019) 
Link: https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/addresses/addresses-darklist.json
"""
def get_additional_scam_addresses():
    address_darklist = json.loads(open('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/illegal_lists/addresses-darklist.json').read())
    print("Number of illegal addresses: ", len(address_darklist))
    addresses, comments, date = ([] for i in range(3))
    for item in address_darklist:
        addresses.append(item['address'])
        comments.append(item['comment'])
        date.append(item['date'])
    print(addresses[0], " ", comments[0], " ", date[0])


"""
Scope: Additional documented scam/illicit behavior URLs
Method Info: Parse local json file and return as array
Total Number of Available documented scam URLs: 2370 (19/02/2019) 
Link: https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/urls/urls-darklist.json
"""
def get_additional_scam_websites():
    url_darklist = json.loads(open('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/illegal_lists/urls-darklist.json', encoding="utf8").read())
    print("Number of illegal addresses: ", len(url_darklist))
    url, comments = ([] for i in range(2))
    for item in url_darklist:
        url.append(item['id'])
        comments.append(item['comment'])
    print(url[0], " ", comments[0])


"""
Scope: Get verified source code from Etherscan using API key 
Method Info: Generate request to retrieve source code for verified contracts from etherscan
API Key: "1BDEBF8IZY2H7ENVHPX6II5ZHEBIJ8V33N"
Link: https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/urls/urls-darklist.json
"""
def get_Transactions():
    #https: // api.etherscan.io / api?module = contract & action = getsourcecode & address = 0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413 & apikey = YourApiKeyToken
    return 0

"""
Scope: Get Current Ether Price and supply in terms of USD from etherscan
Method Info: Request price and supply info using Etherscan API
API Key: "1BDEBF8IZY2H7ENVHPX6II5ZHEBIJ8V33N"
Link: https://github.com/corpetty/py-etherscan-api/blob/master/examples/contracts/get_abi.py
"""
def get_Last_Ether_Price_Supply():
    from etherscan.stats import Stats
    with open("api_key.json", mode='r') as key_file:
        key = json.loads(key_file.read())['key']

    api = Stats(api_key=key)
    ether_last_price_json = api.get_ether_last_price()
    ether_btc = ether_last_price_json['ethbtc']
    ether_datetime = convertTimestampToDateTime(ether_last_price_json['ethbtc_timestamp'])
    ether_usd_price = ether_last_price_json['ethusd']
    #ether_usd_price_datetime = convertTimestampToDateTime(ether_last_price_json['ethusd_timestamp'])
    total_ether_supply = api.get_total_ether_supply()
    print("Time of price: ", ether_datetime, " Ether_BTC price: ", ether_btc, " Ether_USD price: ", ether_usd_price)
    print("Total Ether supply available: ", total_ether_supply)

def convertTimestampToDateTime(timestampValue):
    timestampValue = int(timestampValue)
    value = datetime.datetime.fromtimestamp(timestampValue)
    exct_time = value.strftime('%d %B %Y %H:%M:%S')
    return exct_time

def main():
    get_scams()
    get_scam_addresses()
    get_additional_scam_addresses()
    get_additional_scam_websites()
    get_Last_Ether_Price_Supply()

if __name__ == '__main__':
    main()
