# Created on 19.02.2019 By Steven Farrugia
# Scope: Retrieve and analyze fraudulent activity on the Ethereum network
# Method of retrieval: EtherscamDB API; https://etherscamdb.info/api, MyEtherWallet; https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/addresses/addresses-darklist.json
import datetime
import json
import requests
import numpy as np

"""
Scope: Get all documented scams using etherscamDB API
Method Info: Parse response as json file and retain relevant fields on which analysis shall be carried out
"""
def get_illicit_account_addresses():
    # Total Number of Available documented scams: 6434 (19/02/2019)
    # Total Number of Available documented scams: 6563 (08/04/2019)
    response = requests.get("https://etherscamdb.info/api/scams/")
    if response.status_code == 200:
        response = response.json()
        no_of_scams = len(response['result'])
        scam_id, scam_name, scam_status, scam_category, addresses= ([] for i in range(5))

        for scam in range(no_of_scams):
            if 'addresses' in response['result'][scam]:
                for i in response['result'][scam]['addresses']:
                    addresses.append(i)
                    scam_id.append(response['result'][scam]['id'])
                    scam_name.append(response['result'][scam]['name'])
                    scam_status.append(response['result'][scam]['status'])
                    if 'category' in response['result'][scam]:
                        scam_category.append(response['result'][scam]['category'])
                    else:
                        scam_category.append('Null')
        print("file number of illicit accounts: ", len(addresses))
        print("Unique illicit accounts: ", len(np.unique(addresses)))

        # Total Number of Available documented scam addresses: 692 (19/02/2019)
        # JSON File
        address_darklist = json.loads(open('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Etherlists_And_Stats/illegal_lists/addresses-darklist.json').read())
        addresses_2 = []
        for item in address_darklist:
            addresses_2.append(item['address'])
        print("Number of illegal addresses: ", len(address_darklist))
        print("Number of unique illegal addresses in JSON file: ", len(np.unique(addresses_2)))

        all_addresses = []
        all_addresses = np.concatenate((addresses, addresses_2), axis=None)
        all_addresses = np.unique(all_addresses)
        print("Final number of unique Addresses: ", len(np.unique(all_addresses)))
        return all_addresses



"""
Scope: Additional documented scam/illicit behavior URLs
Method Info: Parse local json file and return as array
Total Number of Available documented scam URLs: 2370 (19/02/2019) 
Link: https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/urls/urls-darklist.json
"""
def get_additional_scam_websites():
    url_darklist = json.loads(open('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Etherlists_And_Stats/illegal_lists/urls-darklist.json', encoding="utf8").read())
    print("Number of illegal addresses: ", len(url_darklist))
    url, comments = ([] for i in range(2))
    for item in url_darklist:
        url.append(item['id'])
        comments.append(item['comment'])
    print(url[0], " ", comments[0])


"""
Scope: Get Current Ether Price and supply in terms of USD from etherscan
Method Info: Request price and supply info using Etherscan API
API Key: "1BDEBF8IZY2H7ENVHPX6II5ZHEBIJ8V33N"
Link: https://github.com/corpetty/py-etherscan-api/blob/master/examples/contracts/get_abi.py
"""
def get_Last_Ether_Price_Supply():
    from etherscan.stats import Stats
    with open("Etherlists_And_Stats/api_key.json", mode='r') as key_file:
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
    addresses = get_illicit_account_addresses()
    #get_additional_scam_websites()
    #get_Last_Ether_Price_Supply()

if __name__ == '__main__':
    main()
